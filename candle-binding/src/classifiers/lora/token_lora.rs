//! LoRA Token Classification

use crate::core::config_errors;
use crate::core::unified_error::{ErrorUnification, ModelErrorType};
use crate::model_architectures::lora::lora_adapter::{LoRAAdapter, LoRAConfig};
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer;

// Import unified tokenization system
use crate::core::tokenization::{create_lora_compatibility_tokenizer, DualPathTokenizer};

/// LoRA Token Classification Result
#[derive(Debug, Clone)]
pub struct LoRATokenResult {
    pub token: String,
    pub label_id: usize,
    pub label_name: String,
    pub confidence: f32,
    pub start_pos: usize,
    pub end_pos: usize,
}

/// LoRA Token Classifier for token-level classification tasks
pub struct LoRATokenClassifier {
    /// BERT model for generating embeddings
    bert: BertModel,
    /// LoRA adapters for different token classification tasks
    adapters: HashMap<String, LoRAAdapter>,
    /// Base token classifier
    base_classifier: candle_nn::Linear,
    /// Unified tokenizer compatible with dual-path architecture
    tokenizer: Box<dyn DualPathTokenizer>,
    /// Computing device
    device: Device,
    /// Label mappings (id -> label_name)
    id2label: HashMap<usize, String>,
    /// Label mappings (label_name -> id)
    label2id: HashMap<String, usize>,
    /// Confidence threshold for predictions
    confidence_threshold: f32,
    /// Hidden size of the model
    hidden_size: usize,
    /// BERT configuration
    config: Config,
}

impl LoRATokenClassifier {
    /// Create new LoRA token classifier from model path
    pub fn new(model_path: &str, use_cpu: bool) -> Result<Self> {
        let device = if use_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0)?
        };

        // Load model configuration using unified config loader
        let token_config = Self::load_token_config(model_path)?;
        let id2label = token_config.id2label;
        let label2id = token_config.label2id;
        let num_labels = token_config.num_labels;
        let hidden_size = token_config.hidden_size;

        // Load BERT configuration
        let config_path = Path::new(model_path).join("config.json");
        let config_str = std::fs::read_to_string(&config_path).map_err(|_e| {
            let unified_err = config_errors::file_not_found(&config_path.to_string_lossy());
            candle_core::Error::from(unified_err)
        })?;
        let config: Config = serde_json::from_str(&config_str).map_err(|e| {
            let unified_err =
                config_errors::invalid_json(&config_path.to_string_lossy(), &e.to_string());
            candle_core::Error::from(unified_err)
        })?;

        // Load tokenizer
        let tokenizer_path = Path::new(model_path).join("tokenizer.json");
        let base_tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|_e| {
            let unified_err = config_errors::file_not_found(&tokenizer_path.to_string_lossy());
            candle_core::Error::from(unified_err)
        })?;

        // Create LoRA-compatible tokenizer
        let tokenizer = create_lora_compatibility_tokenizer(base_tokenizer, device.clone())
            .with_model_context(
                ModelErrorType::Tokenizer,
                "create_lora_compatibility_tokenizer",
                None,
            )
            .map_err(|unified_err| candle_core::Error::from(unified_err))?;

        // Load LoRA configuration
        let lora_config_path = Path::new(model_path).join("lora_config.json");
        let lora_config_content = std::fs::read_to_string(&lora_config_path).map_err(|_e| {
            let unified_err = config_errors::file_not_found(&lora_config_path.to_string_lossy());
            candle_core::Error::from(unified_err)
        })?;

        let lora_config_json: serde_json::Value = serde_json::from_str(&lora_config_content)
            .map_err(|e| {
                let unified_err = config_errors::invalid_json(
                    &lora_config_path.to_string_lossy(),
                    &e.to_string(),
                );
                candle_core::Error::from(unified_err)
            })?;

        let _lora_config = LoRAConfig {
            rank: lora_config_json
                .get("rank")
                .and_then(|v| v.as_u64())
                .unwrap_or(16) as usize,
            alpha: lora_config_json
                .get("alpha")
                .and_then(|v| v.as_f64())
                .unwrap_or(32.0),
            dropout: lora_config_json
                .get("dropout")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.1),
            target_modules: lora_config_json
                .get("target_modules")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(|s| s.to_string()))
                        .collect()
                })
                .unwrap_or_else(|| vec!["classifier".to_string()]),
            use_bias: true,
            ..Default::default()
        };

        // Initialize model weights
        let weights_path = Path::new(model_path).join("model.safetensors");
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? };

        // Load BERT model
        let bert = BertModel::load(vb.pp("bert"), &config)?;

        // Create base classifier
        let base_classifier = linear(hidden_size, num_labels, vb.pp("classifier"))?;

        // For merged LoRA models, we don't need separate adapters
        // The LoRA weights have already been merged into the base classifier
        let adapters = HashMap::new();

        println!("  Using merged LoRA model (no separate adapters needed)");

        Ok(Self {
            bert,
            adapters,
            base_classifier,
            tokenizer,
            device,
            id2label,
            label2id,
            confidence_threshold: 0.5,
            hidden_size,
            config,
        })
    }

    /// Load token configuration from model config.json using unified config loader
    fn load_token_config(model_path: &str) -> Result<crate::core::config_loader::TokenConfig> {
        use crate::core::config_loader::{ConfigLoader, TokenConfigLoader};
        use std::path::Path;

        let path = Path::new(model_path);
        TokenConfigLoader::load_from_path(path)
            .map_err(|unified_err| candle_core::Error::from(unified_err))
    }

    /// Classify tokens in text using LoRA-enhanced model
    pub fn classify_tokens(&self, text: &str) -> Result<Vec<LoRATokenResult>> {
        let start_time = Instant::now();

        // Use real tokenization and classification based on model configuration
        let tokens = self.tokenize_with_bert_compatible(text)?;

        // Check if labels use BIO format
        let is_bio_format = self
            .id2label
            .values()
            .any(|v| v.starts_with("B-") || v.starts_with("I-"));

        // Collect per-token predictions with real offsets
        struct TokenPrediction {
            label_name: String,
            label_id: usize,
            confidence: f32,
            start: usize,
            end: usize,
        }
        let mut predictions = Vec::new();

        for (_i, (token, token_embedding, start, end)) in tokens.iter().enumerate() {
            // Skip special tokens (zero-length offsets)
            if *start == 0 && *end == 0 {
                continue;
            }

            // Add batch dimension: [hidden_size] -> [1, hidden_size]
            let token_embedding_batched = token_embedding.unsqueeze(0)?;

            // Apply base classifier
            let base_logits = self.base_classifier.forward(&token_embedding_batched)?;

            // Apply LoRA adapters if available
            let enhanced_logits = if let Some(adapter) = self.adapters.get("token_classification") {
                let adapter_output = adapter.forward(&token_embedding_batched, false)?;
                (&base_logits + &adapter_output)?
            } else {
                base_logits
            };

            // Apply softmax to get probabilities and remove batch dimension
            let probabilities = candle_nn::ops::softmax(&enhanced_logits, 1)?;
            let probs_vec = probabilities.squeeze(0)?.to_vec1::<f32>()?;

            // Find the class with highest probability
            let (predicted_id, confidence) = probs_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, &conf)| (idx, conf))
                .unwrap_or((0, 0.0));

            if confidence > self.confidence_threshold {
                let label_name = self
                    .id2label
                    .get(&predicted_id)
                    .cloned()
                    .unwrap_or_else(|| format!("LABEL_{}", predicted_id));

                let token_text = if *start < text.len() && *end <= text.len() {
                    &text[*start..*end]
                } else {
                    "<out-of-bounds>"
                };
                println!(
                    "[DEBUG LoRA] token={:?} label={} (id={}) conf={:.4} offset=[{}:{}]",
                    token_text, label_name, predicted_id, confidence, start, end
                );

                predictions.push(TokenPrediction {
                    label_name,
                    label_id: predicted_id,
                    confidence,
                    start: *start,
                    end: *end,
                });
            }
        }

        println!("[DEBUG LoRA] is_bio_format={}, total predictions={}", is_bio_format, predictions.len());

        // Entity merging
        let mut results = Vec::new();

        if is_bio_format {
            // BIO entity merging (same pattern as ModernBERT)
            struct CurrentEntity {
                entity_type: String,
                label_id: usize,
                start: usize,
                end: usize,
                confidence: f32,
            }
            let mut current_entity: Option<CurrentEntity> = None;

            for pred in &predictions {
                if pred.label_name.starts_with("B-") {
                    // Push previous entity
                    if let Some(entity) = current_entity.take() {
                        let entity_text = if entity.start < text.len() && entity.end <= text.len() {
                            text[entity.start..entity.end].to_string()
                        } else {
                            String::new()
                        };
                        results.push(LoRATokenResult {
                            token: entity_text,
                            label_id: entity.label_id,
                            label_name: entity.entity_type.clone(),
                            confidence: entity.confidence,
                            start_pos: entity.start,
                            end_pos: entity.end,
                        });
                    }
                    let entity_type = pred.label_name[2..].to_string();
                    current_entity = Some(CurrentEntity {
                        entity_type,
                        label_id: pred.label_id,
                        start: pred.start,
                        end: pred.end,
                        confidence: pred.confidence,
                    });
                } else if let Some(entity_type) = pred.label_name.strip_prefix("I-") {
                    if let Some(ref mut entity) = current_entity {
                        if entity.entity_type == entity_type {
                            entity.end = pred.end;
                            entity.confidence = (entity.confidence + pred.confidence) / 2.0;
                        } else {
                            // Different type, push current
                            let entity_text = if entity.start < text.len() && entity.end <= text.len() {
                                text[entity.start..entity.end].to_string()
                            } else {
                                String::new()
                            };
                            results.push(LoRATokenResult {
                                token: entity_text,
                                label_id: entity.label_id,
                                label_name: entity.entity_type.clone(),
                                confidence: entity.confidence,
                                start_pos: entity.start,
                                end_pos: entity.end,
                            });
                            current_entity = None;
                        }
                    }
                } else {
                    // O tag or other
                    if let Some(entity) = current_entity.take() {
                        let entity_text = if entity.start < text.len() && entity.end <= text.len() {
                            text[entity.start..entity.end].to_string()
                        } else {
                            String::new()
                        };
                        results.push(LoRATokenResult {
                            token: entity_text,
                            label_id: entity.label_id,
                            label_name: entity.entity_type.clone(),
                            confidence: entity.confidence,
                            start_pos: entity.start,
                            end_pos: entity.end,
                        });
                    }
                }
            }
            // Push final entity
            if let Some(entity) = current_entity.take() {
                let entity_text = if entity.start < text.len() && entity.end <= text.len() {
                    text[entity.start..entity.end].to_string()
                } else {
                    String::new()
                };
                results.push(LoRATokenResult {
                    token: entity_text,
                    label_id: entity.label_id,
                    label_name: entity.entity_type.clone(),
                    confidence: entity.confidence,
                    start_pos: entity.start,
                    end_pos: entity.end,
                });
            }
        } else {
            // Simple adjacent-class merging for non-BIO models
            let mut i = 0;
            while i < predictions.len() {
                let pred = &predictions[i];
                if pred.label_name == "O" || pred.label_id == 0 {
                    i += 1;
                    continue;
                }
                // Start a new merged entity
                let mut end = pred.end;
                let mut confidence = pred.confidence;
                let mut count = 1;
                let label_id = pred.label_id;
                let label_name = pred.label_name.clone();
                let start = pred.start;

                // Merge consecutive tokens with same class
                while i + 1 < predictions.len()
                    && predictions[i + 1].label_id == label_id
                {
                    i += 1;
                    end = predictions[i].end;
                    confidence += predictions[i].confidence;
                    count += 1;
                }

                let entity_text = if start < text.len() && end <= text.len() {
                    text[start..end].to_string()
                } else {
                    String::new()
                };
                results.push(LoRATokenResult {
                    token: entity_text,
                    label_id,
                    label_name,
                    confidence: confidence / count as f32,
                    start_pos: start,
                    end_pos: end,
                });
                i += 1;
            }
        }

        let duration = start_time.elapsed();
        println!(
            "LoRA token classification completed: {} tokens in {:?}",
            results.len(),
            duration
        );
        for r in &results {
            println!(
                "[DEBUG LoRA merged] entity={:?} label={} (id={}) conf={:.4} pos=[{}:{}]",
                r.token, r.label_name, r.label_id, r.confidence, r.start_pos, r.end_pos
            );
        }

        Ok(results)
    }

    /// BERT-compatible tokenization with embeddings and character offsets
    fn tokenize_with_bert_compatible(&self, text: &str) -> Result<Vec<(String, Tensor, usize, usize)>> {
        // Use real BERT tokenization through unified tokenizer
        let tokenization_result = self
            .tokenizer
            .tokenize_for_lora(text)
            .with_model_context(ModelErrorType::Tokenizer, "tokenize_for_lora", Some(text))
            .map_err(|unified_err| candle_core::Error::from(unified_err))?;

        // Clone tokens and offsets before creating tensors to avoid borrow checker issues
        let token_strings = tokenization_result.tokens.clone();
        let offsets = tokenization_result.offsets.clone();
        let (token_ids_tensor, attention_mask_tensor) = self
            .tokenizer
            .create_tensors(&tokenization_result)
            .with_processing_context("create_tensors", Some("token_lora"))
            .map_err(|unified_err| candle_core::Error::from(unified_err))?;

        // Create token type IDs (all zeros for single sentence)
        let token_type_ids = token_ids_tensor.zeros_like()?;

        // Forward pass through BERT to get token-level embeddings
        let hidden_states = self.bert.forward(
            &token_ids_tensor,
            &token_type_ids,
            Some(&attention_mask_tensor),
        )?;

        // Extract token-level embeddings (shape: [batch_size, seq_len, hidden_size])
        // Remove batch dimension since we're processing single text
        let token_embeddings = hidden_states.squeeze(0)?; // Shape: [seq_len, hidden_size]

        // Create result vector with token strings, embeddings, and character offsets
        let mut results = Vec::new();
        let seq_len = token_strings.len();

        for (i, token) in token_strings.iter().enumerate() {
            if i < seq_len && i < offsets.len() {
                // Extract embedding for this token
                let token_embedding = token_embeddings.i(i)?; // Shape: [hidden_size]
                let (start, end) = offsets[i];
                results.push((token.clone(), token_embedding, start, end));
            }
        }

        Ok(results)
    }

    /// Generate contextual embedding based on word content
    fn generate_contextual_embedding(&self, word: &str) -> Result<Tensor> {
        // Use real BERT model to generate contextual embeddings

        // Tokenize the word using our unified tokenizer
        let tokenization_result = self
            .tokenizer
            .tokenize_for_lora(word)
            .with_model_context(ModelErrorType::Tokenizer, "tokenize_for_lora", Some(word))
            .map_err(|unified_err| candle_core::Error::from(unified_err))?;
        let (token_ids_tensor, attention_mask_tensor) = self
            .tokenizer
            .create_tensors(&tokenization_result)
            .with_processing_context("create_tensors", Some("generate_contextual_embedding"))
            .map_err(|unified_err| candle_core::Error::from(unified_err))?;

        // Create token type IDs (all zeros for single sentence)
        let token_type_ids = token_ids_tensor.zeros_like()?;

        // Forward pass through BERT
        let hidden_states = self.bert.forward(
            &token_ids_tensor,
            &token_type_ids,
            Some(&attention_mask_tensor),
        )?;

        // For single word, we can use mean pooling over all tokens
        // or just take the CLS token embedding, or the first non-special token

        // Option 1: Mean pooling (excluding special tokens)
        let seq_len = hidden_states.dim(1)?;
        if seq_len <= 2 {
            // Only CLS and SEP tokens, use CLS token
            let cls_embedding = hidden_states.i((.., 0))?; // CLS token
            return Ok(cls_embedding.squeeze(0)?);
        }

        // Mean pooling over actual word tokens (excluding CLS and SEP)
        let word_embeddings = hidden_states.i((.., 1..seq_len - 1))?; // Exclude CLS and SEP
        let mean_embedding = word_embeddings.mean(1)?; // Mean over sequence dimension

        Ok(mean_embedding.squeeze(0)?) // Remove batch dimension
    }

    /// Get label name from ID
    pub fn get_label_name(&self, label_id: usize) -> Option<&String> {
        self.id2label.get(&label_id)
    }

    /// Get label ID from name
    pub fn get_label_id(&self, label_name: &str) -> Option<usize> {
        self.label2id.get(label_name).copied()
    }

    /// Get all available labels
    pub fn get_all_labels(&self) -> Vec<&String> {
        let mut labels: Vec<_> = self.id2label.values().collect();
        labels.sort();
        labels
    }
}

impl std::fmt::Debug for LoRATokenClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoRATokenClassifier")
            .field("device", &self.device)
            .field("num_labels", &self.id2label.len())
            .field("hidden_size", &self.hidden_size)
            .field("confidence_threshold", &self.confidence_threshold)
            .finish()
    }
}
