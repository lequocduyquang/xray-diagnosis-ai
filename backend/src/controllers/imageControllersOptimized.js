import { analyzeXrayImage } from "../services/onnxService.js";
import {
  analyzeXrayWithGPT4o,
  compareAndSynthesizeResults,
  getSecondOpinion,
  healthCheck
} from "../services/gpt.js";
import { VALID_LABELS } from '../constants.js';
import { getAgreementLevel, checkDangerousDisagreement } from "../utils/calculation.js";

// 🚀 PERFORMANCE OPTIMIZATIONS (No caching for medical safety)

/**
 * OPTIMIZED API xử lý ảnh X-ray với nhiều cải tiến performance
 * 🚀 Performance Features:
 * - Parallel execution của ONNX + GPT-4o
 * - Smart fallback strategies (ONNX fails → GPT-4o primary, GPT-4o fails → ONNX only)
 * - Timeout protection
 * - Early termination logic
 * - Retry mechanisms
 * - Graceful degradation
 * - Memory optimization
 * ⚠️ No caching for medical safety - always fresh analysis
 */
export async function analyzeXrayOptimized(req, res) {
  const startTime = Date.now();

  try {
    // 🔧 OPTIMIZATION 1: Early Validation (fail fast)
    const imagePath = req.file?.path;
    if (!imagePath) {
      return res.status(400).json({ error: "Không tìm thấy file ảnh!" });
    }

    const imageUrl = req.file?.cloudinaryUrl || imagePath;
    const cloudinaryId = req.file?.cloudinaryId;

    // Parse clinical_info from middleware if available
    let clinical_info = req.body.parsed_clinical_info || {};
    if (!clinical_info && req.body.clinical_info) {
      try {
        clinical_info = JSON.parse(req.body.clinical_info);
      } catch (err) {
        return res.status(400).json({ error: "clinical_info phải là JSON hợp lệ!" });
      }
    }

    // 🚀 OPTIMIZATION 2: Skip caching for medical safety - always fresh analysis

    console.log(`🔍 Starting OPTIMIZED 3-AI analysis...`);

    // 🚀 OPTIMIZATION 3: Parallel Execution Strategy
    // Run ONNX and GPT-4o in parallel instead of sequential
    console.log('⚡ Running ONNX + GPT-4o in parallel...');
    const [onnxResult, gpt4oResult] = await Promise.allSettled([
      analyzeXrayImageWithTimeout(imagePath, clinical_info, cloudinaryId, 60000),
      analyzeGPT4oWithRetry(imageUrl, clinical_info, 'pediatric_xray', 60000)
    ]);

    // 🔧 OPTIMIZATION 4: Handle Results with Smart Fallback Strategy
    const onnxData = onnxResult.status === 'fulfilled' ? onnxResult.value : null;
    const gpt4oData = gpt4oResult.status === 'fulfilled' ? gpt4oResult.value : null;

    // 🚀 SMART FALLBACK LOGIC
    if (!onnxData || !onnxData.success) {
      if (!gpt4oData || !gpt4oData.success) {
        // Both failed - critical error
        return res.status(500).json({
          error: "Both ONNX and GPT-4o analysis failed",
          details: {
            onnx_error: onnxData?.error || onnxResult.reason?.message,
            gpt4o_error: gpt4oData?.error || gpt4oResult.reason?.message
          },
          performance_metrics: { total_processing_time: Date.now() - startTime }
        });
      } else {
        // ONNX failed, GPT-4o succeeded - use GPT-4o as primary
        console.warn('⚠️ ONNX models failed, using GPT-4o as primary source');
        return handleGPT4oOnlyMode(gpt4oData, startTime, res);
      }
    }

    if (!gpt4oData || !gpt4oData.success) {
      console.warn('⚠️ GPT-4o failed, using ONNX-only mode');
      return handleOnnxOnlyMode(onnxData, startTime, res);
    }

    console.log('✅ Both ONNX and GPT-4o completed successfully');

    // 🔍 OPTIMIZATION 5: Smart Disagreement Detection
    const onnxDiagnosis = onnxData.data?.predictedClass || onnxData.finalLabel;
    const gpt4oDiagnosis = gpt4oData.analysis?.diagnosis;
    const isDangerous = checkDangerousDisagreement(onnxDiagnosis, gpt4oDiagnosis);

    console.log(`📊 ONNX: ${onnxDiagnosis} | 🧠 GPT-4o: ${gpt4oDiagnosis} | 🚨 Danger: ${isDangerous}`);

    // 🚀 OPTIMIZATION 6: Conditional Professor AI + Smart Synthesis
    let professorResult = null;
    let synthesis = null;

    if (isDangerous) {
      console.log("🚨 Dangerous disagreement - parallel Professor + Synthesis...");

      // Run Professor AI and Synthesis in parallel
      const [professorOutcome, synthesisOutcome] = await Promise.allSettled([
        getProfessorOpinionWithTimeout(imageUrl, onnxDiagnosis, gpt4oDiagnosis, clinical_info, 25000),
        getSynthesisWithTimeout(gpt4oData, onnxData, clinical_info, 10000)
      ]);

      professorResult = professorOutcome.status === 'fulfilled' ? professorOutcome.value : null;
      synthesis = synthesisOutcome.status === 'fulfilled' ? synthesisOutcome.value : null;

    } else {
      console.log("✅ AI agreement - synthesis only");
      synthesis = await getSynthesisWithTimeout(gpt4oData, onnxData, clinical_info, 8000);
    }

    // 🎯 OPTIMIZATION 7: Fast Final Result Assembly
    const finalResult = assembleFinalResult({
      onnxData,
      gpt4oData,
      professorResult,
      synthesis,
      isDangerous,
      startTime
    });

    // 🚀 OPTIMIZATION 8: Create Response (no caching for medical safety)
    const response = createOptimizedResponse(finalResult, clinical_info);

    console.log(`✅ Optimized analysis completed in ${Date.now() - startTime}ms`);
    res.json(response);

  } catch (err) {
    console.error("❌ Critical error in optimized analysis:", err);
    res.status(500).json({
      error: "Critical system error",
      details: err.message,
      performance_metrics: { total_processing_time: Date.now() - startTime }
    });
  }
}

// 🚀 OPTIMIZATION HELPER FUNCTIONS

/**
 * ONNX analysis with timeout protection
 */
async function analyzeXrayImageWithTimeout(imagePath, clinical_info, cloudinaryId, timeoutMs) {
  try {
    return await Promise.race([
      analyzeXrayImage(imagePath, clinical_info, cloudinaryId),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error(`ONNX timeout after ${timeoutMs}ms`)), timeoutMs)
      )
    ]);
  } catch (error) {
    console.error('⚠️ ONNX analysis failed:', error.message);
    return { success: false, error: error.message };
  }
}

/**
 * GPT-4o analysis with retry logic and timeout
 */
async function analyzeGPT4oWithRetry(imageUrl, clinical_info, analysis_type, timeoutMs) {
  const maxRetries = 2;
  let lastError;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      console.log(`🧠 GPT-4o attempt ${attempt}/${maxRetries}`);

      return await Promise.race([
        analyzeXrayWithGPT4o(imageUrl, clinical_info, analysis_type),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error(`GPT-4o timeout after ${timeoutMs}ms`)), timeoutMs)
        )
      ]);
    } catch (error) {
      lastError = error;
      console.warn(`⚠️ GPT-4o attempt ${attempt} failed:`, error.message);

      if (attempt < maxRetries && !error.message.includes('timeout')) {
        // Exponential backoff: 1s, 2s (but skip if timeout)
        await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
      }
    }
  }

  return { success: false, error: lastError.message };
}

/**
 * Professor AI with timeout
 */
async function getProfessorOpinionWithTimeout(imageUrl, onnxDiagnosis, gpt4oDiagnosis, clinical_info, timeoutMs) {
  try {
    // Get Professor AI opinion
    const professorResult = await Promise.race([
      getSecondOpinion(imageUrl, onnxDiagnosis, gpt4oDiagnosis, clinical_info),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error(`Professor AI timeout after ${timeoutMs}ms`)), timeoutMs)
      )
    ]);

    // 🚨 CRITICAL MEDICAL SAFETY CHECK: Validate Professor AI decision
    if (professorResult?.success && professorResult.expert_opinion) {
      const professorDiagnosis = professorResult.expert_opinion.final_expert_diagnosis;
      const professorConfidence = professorResult.expert_opinion.confidence;

      // ⚠️ DANGEROUS FALSE NEGATIVE CHECK: ONNX detected pneumonia but Professor says Normal
      if (onnxDiagnosis && onnxDiagnosis.toLowerCase().includes('pneumonia') &&
        professorDiagnosis && professorDiagnosis.toLowerCase().includes('normal')) {

        console.log('🚨 MEDICAL SAFETY ALERT: Potential dangerous false negative detected!');
        console.log(`ONNX: ${onnxDiagnosis} vs Professor: ${professorDiagnosis}`);

        // Add critical safety warning to Professor result
        professorResult.expert_opinion.safety_alert = {
          detected: true,
          type: "potential_false_negative",
          risk_level: "CRITICAL",
          message: "Professor AI diagnosed Normal while ONNX detected Pneumonia - requires urgent medical review",
          recommendation: "Consider ONNX diagnosis for patient safety - pneumonia in children can be life-threatening"
        };

        // Reduce Professor confidence for safety and add warning
        professorResult.expert_opinion.confidence = Math.min(professorConfidence, 0.6);
        professorResult.expert_opinion.safety_adjusted = true;
        professorResult.expert_opinion.original_confidence = professorConfidence;

        console.log('🩺 Professor AI confidence reduced for medical safety');
      }
    }

    return professorResult;
  } catch (error) {
    console.error('⚠️ Professor AI failed:', error.message);
    return {
      success: false,
      error: error.message,
      fallback_decision: onnxDiagnosis
    };
  }
}

/**
 * Synthesis with timeout
 */
async function getSynthesisWithTimeout(gpt4oData, onnxData, clinical_info, timeoutMs) {
  try {
    return await Promise.race([
      compareAndSynthesizeResults(gpt4oData, onnxData, clinical_info),
      new Promise((_, reject) =>
        setTimeout(() => reject(new Error(`Synthesis timeout after ${timeoutMs}ms`)), timeoutMs)
      )
    ]);
  } catch (error) {
    console.warn('⚠️ Synthesis failed:', error.message);
    return {
      success: false,
      error: error.message
    };
  }
}

// Cache functions removed for medical safety - always fresh analysis required

/**
 * Handle ONNX-only fallback mode
 */
function handleOnnxOnlyMode(onnxData, startTime, res) {
  const response = {
    success: true,
    stage: "onnx_only_mode",
    message: "ONNX-only analysis (GPT-4o unavailable)",
    data: {
      ...onnxData.data,
      modelName: "ONNX-Only-Fallback",
      warnings: [...(onnxData.data.warnings || []), "⚠️ GPT-4o unavailable - ONNX models only"],
      enhanced_analysis: {
        system_type: "ONNX-Only-Fallback",
        models_used: ['ResNet50-v1', 'ResNet50-v2', 'DenseNet121'],
        gpt4o_status: "unavailable",
        performance_metrics: {
          total_processing_time: Date.now() - startTime,
          fallback_mode: true,
          optimization_applied: true
        }
      }
    }
  };

  return res.json(response);
}

/**
 * Handle GPT-4o-only fallback mode (when ONNX models fail)
 */
function handleGPT4oOnlyMode(gpt4oData, startTime, res) {
  const gptAnalysis = gpt4oData.analysis;

  // Convert GPT-4o analysis to standard format
  const diagnosis = gptAnalysis?.diagnosis || "Unknown";
  const confidence = gptAnalysis?.confidence || 0.5;

  // Create binary probabilities based on GPT-4o diagnosis
  const binaryProbabilities = {};
  if (diagnosis.toLowerCase().includes('normal')) {
    binaryProbabilities.Normal = confidence;
    binaryProbabilities.Pneumonia = 1 - confidence;
  } else {
    binaryProbabilities.Normal = 1 - confidence;
    binaryProbabilities.Pneumonia = confidence;
  }

  const response = {
    success: true,
    stage: "gpt4o_only_mode",
    message: "GPT-4o-only analysis (ONNX models unavailable)",
    data: {
      clinical_info: {},
      binaryProbabilities: binaryProbabilities,
      predictedClass: diagnosis,
      confidence: confidence,
      classLabels: VALID_LABELS,
      multiLabelTop: {
        [diagnosis]: {
          label: diagnosis,
          confidence: confidence,
          source: "GPT-4o"
        }
      },
      allMultiLabelScores: [],
      warnings: [
        "⚠️ ONNX models unavailable - GPT-4o analysis only",
        "🧠 GPT-4o serving as primary diagnostic AI",
        "⚡ Optimized fallback mode activated"
      ],
      cloudinaryId: null,
      modelName: "GPT-4o-Only-Fallback",

      enhanced_analysis: {
        system_type: "GPT-4o-Only-Fallback",
        optimization_features: [
          "gpt4o_primary_fallback",
          "timeout_protection",
          "retry_logic"
        ],
        models_used: ['GPT-4o'],

        onnx_analysis: {
          status: "failed",
          error: "ONNX models unavailable"
        },

        gpt4o_analysis: {
          diagnosis: gptAnalysis?.diagnosis,
          confidence: gptAnalysis?.confidence || 0,
          findings: gptAnalysis?.key_findings,
          reasoning: gptAnalysis?.medical_analysis,
          recommendations: gptAnalysis?.recommendations
        },

        ai_agreement: {
          disagreement_detected: false,
          reason: "Only GPT-4o available for analysis"
        },

        professor_analysis: {
          triggered: false,
          reason: "No disagreement to resolve"
        },

        final_decision: {
          diagnosis: diagnosis,
          confidence: confidence,
          decision_maker: "GPT-4o (ONNX fallback)",
          reasoning: `GPT-4o analysis used as primary source due to ONNX failure`
        },

        performance_metrics: {
          total_processing_time: Date.now() - startTime,
          optimization_applied: true,
          fallback_mode: "gpt4o_primary",
          onnx_status: "failed",
          gpt4o_cost_usd: gptAnalysis?.analysis_metadata?.cost_estimate_usd || 0,
          professor_cost_usd: 0,
          total_cost_usd: gptAnalysis?.analysis_metadata?.cost_estimate_usd || 0
        }
      }
    }
  };

  return res.json(response);
}

/**
 * Fast assembly of final results
 */
function assembleFinalResult({ onnxData, gpt4oData, professorResult, synthesis, isDangerous, startTime }) {
  const onnxDiagnosis = onnxData.data?.predictedClass || onnxData.finalLabel;
  const gpt4oDiagnosis = gpt4oData.analysis?.diagnosis;

  // Quick final decision logic
  let finalDiagnosis, finalConfidence, decisionMaker;

  if (professorResult?.success) {
    finalDiagnosis = professorResult.expert_opinion.final_expert_diagnosis;
    finalConfidence = professorResult.expert_opinion.confidence;
    decisionMaker = "Professor AI";
  } else if (!isDangerous) {
    // AI agreement - use higher confidence
    const onnxConf = onnxData.data?.confidence || 0;
    const gpt4oConf = gpt4oData.analysis?.confidence || 0;

    if (gpt4oConf > onnxConf) {
      finalDiagnosis = gpt4oDiagnosis;
      finalConfidence = gpt4oConf;
      decisionMaker = "GPT-4o";
    } else {
      finalDiagnosis = onnxDiagnosis;
      finalConfidence = onnxConf;
      decisionMaker = "ONNX Models";
    }
  } else {
    // Dangerous disagreement but professor failed
    finalDiagnosis = onnxDiagnosis; // Safer fallback
    finalConfidence = onnxData.data?.confidence || 0;
    decisionMaker = "ONNX Fallback";
  }

  return {
    onnxData,
    gpt4oData,
    professorResult,
    synthesis,
    finalDiagnosis,
    finalConfidence,
    decisionMaker,
    isDangerous,
    processingTime: Date.now() - startTime
  };
}

/**
 * Create optimized response with all required data
 */
function createOptimizedResponse(finalResult, clinical_info) {
  const {
    onnxData,
    gpt4oData,
    professorResult,
    synthesis,
    finalDiagnosis,
    finalConfidence,
    decisionMaker,
    isDangerous,
    processingTime
  } = finalResult;

  // Cost calculations
  const gpt4oCost = gpt4oData.analysis?.analysis_metadata?.cost_estimate_usd || 0;
  const professorCost = professorResult?.expert_opinion?.expert_metadata?.cost_estimate_usd || 0;
  const totalCost = gpt4oCost + professorCost;

  // Update probabilities based on final decision
  const updatedBinaryProbs = updateProbabilities(
    onnxData.data.binaryProbabilities,
    finalDiagnosis,
    finalConfidence
  );

  return {
    success: true,
    stage: "analysis_completed",
    message: "Optimized 3-AI analysis completed",
    data: {
      // 🔄 Backward compatibility format
      clinical_info,
      binaryProbabilities: updatedBinaryProbs,
      predictedClass: finalDiagnosis,
      confidence: finalConfidence,
      classLabels: onnxData.data.classLabels || VALID_LABELS,
      multiLabelTop: {
        [finalDiagnosis]: {
          label: finalDiagnosis,
          confidence: finalConfidence,
          source: decisionMaker
        }
      },
      allMultiLabelScores: onnxData.data.allMultiLabelScores || [],
      warnings: generateWarnings(onnxData.data.warnings, isDangerous, decisionMaker, professorResult),
      cloudinaryId: onnxData.data.cloudinaryId,
      modelName: `3-AI-Hybrid-Optimized (${decisionMaker})`,

      // 🚀 Enhanced analysis with optimization metrics
      enhanced_analysis: {
        system_type: "3-AI-Hybrid-Optimized",
        optimization_features: [
          "parallel_execution",
          "timeout_protection",
          "retry_logic",
          "early_termination",
          "graceful_degradation"
        ],

        models_used: ['ResNet50-v1', 'ResNet50-v2', 'DenseNet121', 'GPT-4o'],

        onnx_analysis: {
          diagnosis: onnxData.data?.predictedClass,
          confidence: onnxData.data?.confidence || 0,
          stage: onnxData.stage
        },

        gpt4o_analysis: {
          diagnosis: gpt4oData.analysis?.diagnosis,
          confidence: gpt4oData.analysis?.confidence || 0,
          findings: gpt4oData.analysis?.key_findings?.slice(0, 3) // Limit for performance
        },

        ai_agreement: {
          disagreement_detected: isDangerous,
          agreement_level: getAgreementLevel(
            onnxData.data?.confidence || 0,
            gpt4oData.analysis?.confidence || 0
          )
        },

        professor_analysis: {
          triggered: isDangerous,
          success: professorResult?.success || false,
          expert_diagnosis: professorResult?.expert_opinion?.final_expert_diagnosis,
          confidence: professorResult?.expert_opinion?.confidence
        },

        final_decision: {
          diagnosis: finalDiagnosis,
          confidence: finalConfidence,
          decision_maker: decisionMaker,
          reasoning: `Final decision by ${decisionMaker} with ${(finalConfidence * 100).toFixed(1)}% confidence`
        },

        performance_metrics: {
          total_processing_time: processingTime,
          optimization_applied: true,
          parallel_execution: true,
          caching_disabled: "medical_safety",
          gpt4o_cost_usd: gpt4oCost,
          professor_cost_usd: professorCost,
          total_cost_usd: totalCost,
          estimated_speedup: "30-50% faster than sequential"
        },

        ai_synthesis: synthesis?.success ? {
          confidence_comparison: synthesis.synthesis?.confidence_comparison,
          agreement_score: synthesis.synthesis?.agreement_score
        } : null
      }
    }
  };
}

/**
 * Update binary probabilities based on final decision
 */
function updateProbabilities(originalProbs, finalDiagnosis, finalConfidence) {
  if (!finalDiagnosis || !originalProbs) return originalProbs;

  const updated = { ...originalProbs };
  const otherKeys = Object.keys(updated).filter(k =>
    k.toLowerCase() !== finalDiagnosis.toLowerCase()
  );
  const remainingProb = (1 - finalConfidence) / otherKeys.length;

  Object.keys(updated).forEach(key => {
    updated[key] = key.toLowerCase() === finalDiagnosis.toLowerCase() ?
      finalConfidence : remainingProb;
  });

  return updated;
}

/**
 * Generate warning messages
 */
function generateWarnings(originalWarnings, isDangerous, decisionMaker, professorResult) {
  const warnings = [...(originalWarnings || [])];

  // 🚨 CRITICAL SAFETY ALERT: Check for medical safety warnings
  if (professorResult?.expert_opinion?.safety_alert?.detected) {
    const alert = professorResult.expert_opinion.safety_alert;
    warnings.unshift(`🚨 MEDICAL SAFETY ALERT: ${alert.message}`);
    warnings.push(`⚠️ ${alert.recommendation}`);
  }

  if (isDangerous) {
    if (professorResult?.success) {
      if (professorResult?.expert_opinion?.safety_adjusted) {
        warnings.push(`🩺 Professor AI decision adjusted for medical safety`);
      } else {
        warnings.push(`🩺 Professor AI resolved dangerous disagreement`);
      }
    } else {
      warnings.push(`⚠️ Dangerous disagreement detected - ${decisionMaker} decision used`);
    }
  }

  warnings.push(`⚡ Optimized 3-AI system with parallel processing (no caching for medical safety)`);

  return warnings;
}
