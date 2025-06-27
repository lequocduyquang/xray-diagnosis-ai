import { analyzeXrayImage } from "../services/onnxService.js";
import {
  analyzeXrayWithGPT4o,
  compareAndSynthesizeResults,
  getSecondOpinion,
  healthCheck
} from "../services/gpt.js";
import { VALID_LABELS } from '../constants.js';
import {
  getAgreementLevel,
  checkDangerousDisagreement,
  getFinalDecisionReasoning,
  generateClinicalRecommendations
} from "../utils/calculation.js";

/**
 * API xử lý ảnh X-ray và trả kết quả phân tích
 * ENHANCED: 3-AI Hybrid System - ONNX models + GPT-4o + Professor AI
 * @param {object} req - Request object (chứa file ảnh và clinical_info)
 * @param {object} res - Response object
 */
export async function analyzeXray(req, res) {
  try {
    const imagePath = req.file?.path;
    if (!imagePath) {
      return res.status(400).json({ error: "Không tìm thấy file ảnh!" });
    }

    // Lấy cloudinary_id từ req.file (đã được thêm trong middleware)
    const cloudinaryId = req.file?.cloudinaryId;
    const imageUrl = req.file?.cloudinaryUrl || imagePath;

    // Parse clinical_info từ body
    let clinical_info = {};
    if (req.body.clinical_info) {
      try {
        clinical_info = JSON.parse(req.body.clinical_info);
      } catch (err) {
        return res
          .status(400)
          .json({ error: "clinical_info phải là JSON hợp lệ!" });
      }
    }

    // Kiểm tra tính hợp lệ của initial_diagnosis (nếu có)
    if (
      clinical_info.initial_diagnosis &&
      !VALID_LABELS.includes(clinical_info.initial_diagnosis)
    ) {
      return res.status(400).json({
        error: `Chẩn đoán lâm sàng không hợp lệ! Phải thuộc: ${VALID_LABELS.join(
          ", "
        )}`,
      });
    }

    // Kiểm tra symptoms (nếu có)
    if (clinical_info.symptoms && !Array.isArray(clinical_info.symptoms)) {
      return res.status(400).json({
        error: "Triệu chứng phải là một mảng (array) các chuỗi!",
      });
    }

    console.log(`🔍 Analyzing X-ray: ${imagePath}`);
    console.log(`📋 Clinical info: ${JSON.stringify(clinical_info)}`);

    // 🚀 ENHANCED: GPT-4o is now ALWAYS enabled for 3-AI system
    console.log(`🤖 3-AI Hybrid System: ONNX + GPT-4o + Professor AI`);

    // Check OpenAI API key (required for new system)
    if (!process.env.OPENAI_API_KEY) {
      return res.status(400).json({
        error: "OPENAI_API_KEY không được cấu hình! Hệ thống 3-AI cần GPT-4o để hoạt động.",
        suggestion: "Thêm OPENAI_API_KEY=your_api_key_here vào file .env"
      });
    }

    // 1. Luôn chạy ONNX models trước (ResNet50 + DenseNet121)
    console.log('🔬 Running ONNX models analysis...');
    const onnxResult = await analyzeXrayImage(
      imagePath,
      clinical_info,
      cloudinaryId
    );

    // 2. 🧠 ALWAYS run GPT-4o analysis (core part of 3-AI system)
    console.log('🧠 Running GPT-4o analysis (required for 3-AI system)...');
    let gpt4oResult;
    let finalResult;

    try {
      gpt4oResult = await analyzeXrayWithGPT4o(
        imageUrl,
        clinical_info,
        'pediatric_xray'
      );

      if (!gpt4oResult.success) {
        return res.status(500).json({
          error: "GPT-4o analysis failed - hệ thống 3-AI không thể hoạt động",
          details: gpt4oResult.error,
          fallback_onnx_result: onnxResult
        });
      }

      console.log('✅ GPT-4o analysis completed');

      // 3. 🔍 CRITICAL: Always check for disagreement between ONNX and GPT-4o
      const onnxDiagnosis = onnxResult.finalLabel;
      const gpt4oDiagnosis = gpt4oResult.analysis?.diagnosis;
      const isDangerous = checkDangerousDisagreement(onnxDiagnosis, gpt4oDiagnosis);

      console.log(`📊 ONNX Diagnosis: ${onnxDiagnosis}`);
      console.log(`🧠 GPT-4o Diagnosis: ${gpt4oDiagnosis}`);
      console.log(`🚨 Disagreement detected: ${isDangerous ? 'YES' : 'NO'}`);

      // Prepare base result
      finalResult = {
        // ONNX Results (backward compatibility)
        ...onnxResult,

        // GPT-4o Results
        gpt4o_analysis: gpt4oResult.analysis,

        // Metadata
        analysis_metadata: {
          timestamp: new Date().toISOString(),
          models_used: ['ResNet50-v1', 'ResNet50-v2', 'DenseNet121', 'GPT-4o'],
          system_type: '3-AI-Hybrid',
          disagreement_detected: isDangerous,
          onnx_diagnosis: onnxDiagnosis,
          gpt4o_diagnosis: gpt4oDiagnosis
        },

        // Confidence comparison
        confidence_comparison: {
          onnx_models: onnxResult.confidence || 0,
          gpt4o: gpt4oResult.analysis?.confidence || 0,
          difference: Math.abs((onnxResult.confidence || 0) - (gpt4oResult.analysis?.confidence || 0)),
          agreement_level: getAgreementLevel(onnxResult.confidence || 0, gpt4oResult.analysis?.confidence || 0)
        },

        // Cost tracking
        cost_tracking: {
          gpt4o_cost_usd: gpt4oResult.analysis?.analysis_metadata?.cost_estimate_usd || 0,
          tokens_used: gpt4oResult.usage?.total_tokens || 0
        }
      };

      // 4. 🩺 If disagreement detected, automatically trigger Professor AI
      if (isDangerous) {
        console.log("🚨 DANGEROUS DISAGREEMENT DETECTED!");
        console.log(`📊 ONNX: ${onnxDiagnosis} vs GPT-4o: ${gpt4oDiagnosis}`);

        try {
          const secondOpinionResult = await getSecondOpinion(
            imageUrl,
            onnxDiagnosis,
            gpt4oDiagnosis,
            clinical_info
          );

          if (secondOpinionResult.success) {
            finalResult.expert_second_opinion = secondOpinionResult.expert_opinion;
            finalResult.analysis_metadata.second_opinion_triggered = true;
            finalResult.analysis_metadata.professor_ai_used = true;

            // 🎯 FINAL DECISION: Professor AI wins!
            finalResult.final_expert_diagnosis = secondOpinionResult.expert_opinion.final_expert_diagnosis;
            finalResult.risk_assessment = secondOpinionResult.expert_opinion.risk_assessment;
            finalResult.urgent_actions = secondOpinionResult.expert_opinion.urgent_actions;
            finalResult.professor_confidence = secondOpinionResult.expert_opinion.confidence;

            // Add professor cost to total
            finalResult.cost_tracking.professor_cost_usd = secondOpinionResult.expert_opinion.expert_metadata?.cost_estimate_usd || 0;
            finalResult.cost_tracking.total_cost_usd =
              (finalResult.cost_tracking.gpt4o_cost_usd || 0) +
              (finalResult.cost_tracking.professor_cost_usd || 0);

            console.log(`🩺 FINAL EXPERT DIAGNOSIS: ${secondOpinionResult.expert_opinion.final_expert_diagnosis}`);
          } else {
            console.error("❌ Expert second opinion failed:", secondOpinionResult.error);
            finalResult.expert_second_opinion = {
              error: "Expert consultation failed - MANUAL REVIEW REQUIRED",
              fallback_action: "IMMEDIATE RADIOLOGIST REVIEW NEEDED"
            };
          }
        } catch (expertError) {
          console.error("❌ Second opinion error:", expertError);
          finalResult.expert_second_opinion = {
            error: "Technical error in expert consultation",
            fallback_action: "URGENT MANUAL REVIEW REQUIRED"
          };
        }
      } else {
        // 5. ✅ No disagreement - both AIs agree
        console.log("✅ ONNX and GPT-4o AGREEMENT detected - no professor needed");
        finalResult.analysis_metadata.ai_agreement = true;
        finalResult.analysis_metadata.second_opinion_triggered = false;

        // When AIs agree, use the higher confidence one
        const onnxConf = onnxResult.confidence || 0;
        const gpt4oConf = gpt4oResult.analysis?.confidence || 0;

        if (gpt4oConf > onnxConf) {
          finalResult.consensus_diagnosis = gpt4oDiagnosis;
          finalResult.consensus_confidence = gpt4oConf;
          finalResult.consensus_source = "GPT-4o (higher confidence)";
        } else {
          finalResult.consensus_diagnosis = onnxDiagnosis;
          finalResult.consensus_confidence = onnxConf;
          finalResult.consensus_source = "ONNX Models (higher confidence)";
        }
      }

      // 6. 🔄 Always run synthesis for comprehensive analysis
      console.log('🔄 Synthesizing AI results...');
      const synthesis = await compareAndSynthesizeResults(
        gpt4oResult,
        onnxResult,
        clinical_info
      );

      finalResult.ai_synthesis = synthesis;
      finalResult.analysis_metadata.synthesis_available = synthesis.success;

    } catch (gpt4oError) {
      console.error('❌ GPT-4o analysis failed:', gpt4oError);
      return res.status(500).json({
        error: "Hệ thống 3-AI gặp lỗi",
        details: gpt4oError.message,
        fallback_onnx_result: onnxResult
      });
    }

    // 7. Generate enhanced clinical recommendations
    finalResult.clinical_recommendations = generateClinicalRecommendations(
      finalResult,
      clinical_info
    );

    // 8. Add system summary
    finalResult.system_summary = {
      total_ai_models: finalResult.analysis_metadata.models_used.length,
      disagreement_resolved: finalResult.analysis_metadata.disagreement_detected && finalResult.analysis_metadata.second_opinion_triggered,
      final_decision_maker: finalResult.analysis_metadata.disagreement_detected ?
        "Professor AI" :
        (finalResult.consensus_source || "AI Consensus"),
      system_confidence: finalResult.professor_confidence ||
        finalResult.consensus_confidence ||
        Math.max(onnxResult.confidence || 0, gpt4oResult.analysis?.confidence || 0)
    };

    console.log('✅ 3-AI Hybrid analysis completed');
    console.log(`🎯 Final decision: ${finalResult.final_expert_diagnosis || finalResult.consensus_diagnosis || onnxResult.finalLabel}`);

    // 🔄 STANDARDIZE RESPONSE FORMAT (backward compatibility with FINAL DECISION)

    // 🎯 GET FINAL DECISION DATA (Professor AI > Consensus > ONNX)
    const finalDiagnosis = finalResult.final_expert_diagnosis ||
      finalResult.consensus_diagnosis ||
      onnxResult.finalLabel;

    const finalConfidence = finalResult.professor_confidence ||
      finalResult.consensus_confidence ||
      (onnxResult.confidence || 0);

    const finalDecisionMaker = finalResult.system_summary?.final_decision_maker || "ONNX Models";

    // 🔄 Create updated binary probabilities based on final decision
    const updatedBinaryProbabilities = { ...onnxResult.binaryProbabilities };
    let updatedMultiLabelTop = { ...onnxResult.multiLabelTop };

    if (finalDiagnosis && finalDiagnosis !== onnxResult.finalLabel) {
      // Update probabilities to reflect final decision
      const otherClasses = Object.keys(updatedBinaryProbabilities).filter(
        key => key.toLowerCase() !== finalDiagnosis.toLowerCase()
      );
      const remainingProb = (1 - finalConfidence) / otherClasses.length;

      Object.keys(updatedBinaryProbabilities).forEach(key => {
        updatedBinaryProbabilities[key] = key.toLowerCase() === finalDiagnosis.toLowerCase()
          ? finalConfidence
          : remainingProb;
      });

      // Update multiLabelTop to reflect final decision
      updatedMultiLabelTop = {
        [finalDiagnosis]: {
          label: finalDiagnosis,
          confidence: finalConfidence,
          source: finalDecisionMaker
        }
      };
    }

    // 🔄 Create warnings based on disagreement status
    const updatedWarnings = [...(onnxResult.warnings || [])];
    if (finalResult.analysis_metadata.disagreement_detected) {
      if (finalResult.analysis_metadata.professor_ai_used) {
        updatedWarnings.push(`🩺 PROFESSOR AI DECISION: ${finalDiagnosis} (resolved dangerous disagreement)`);
      } else {
        updatedWarnings.push(`⚠️ AI DISAGREEMENT: Final decision based on ${finalDecisionMaker}`);
      }
    }

    const standardizedResponse = {
      success: true,
      stage: "analysis_completed",
      message: "3-AI Hybrid analysis completed successfully",
      data: {
        // ✅ UPDATED FORMAT with FINAL DECISION (backward compatibility)
        clinical_info: clinical_info,
        binaryProbabilities: updatedBinaryProbabilities,
        predictedClass: finalDiagnosis,  // 🎯 FINAL DECISION HERE
        confidence: finalConfidence,     // 🎯 FINAL CONFIDENCE HERE
        classLabels: onnxResult.classLabels || VALID_LABELS,
        multiLabelTop: updatedMultiLabelTop,  // 🎯 UPDATED WITH FINAL DECISION
        allMultiLabelScores: onnxResult.allMultiLabelScores || [],
        warnings: updatedWarnings,
        cloudinaryId: onnxResult.cloudinaryId || cloudinaryId,
        modelName: `3-AI-Hybrid-System (Final: ${finalDecisionMaker})`,

        // 🚀 NEW: Enhanced 3-AI System Data
        enhanced_analysis: {
          // System Overview
          system_type: "3-AI-Hybrid",
          models_used: finalResult.analysis_metadata.models_used,
          total_ai_models: finalResult.analysis_metadata.models_used.length,

          // ONNX Results (detailed)
          onnx_analysis: {
            diagnosis: onnxResult.finalLabel,
            confidence: onnxResult.confidence || 0,
            model_details: {
              resnet50_v1: onnxResult.resnet50_v1 || null,
              resnet50_v2: onnxResult.resnet50_v2 || null,
              densenet121: onnxResult.densenet121 || null
            }
          },

          // GPT-4o Results
          gpt4o_analysis: {
            diagnosis: gpt4oResult.analysis?.diagnosis,
            confidence: gpt4oResult.analysis?.confidence || 0,
            reasoning: gpt4oResult.analysis?.medical_analysis,
            findings: gpt4oResult.analysis?.key_findings,
            recommendations: gpt4oResult.analysis?.recommendations,
            risk_factors: gpt4oResult.analysis?.risk_factors
          },

          // AI Agreement Analysis
          ai_agreement: {
            disagreement_detected: finalResult.analysis_metadata.disagreement_detected,
            agreement_level: finalResult.confidence_comparison.agreement_level,
            confidence_difference: finalResult.confidence_comparison.difference,
            consensus_reached: finalResult.analysis_metadata.ai_agreement || false
          },

          // Professor AI (if triggered)
          professor_analysis: finalResult.analysis_metadata.professor_ai_used ? {
            triggered: true,
            expert_diagnosis: finalResult.final_expert_diagnosis,
            expert_confidence: finalResult.professor_confidence,
            risk_assessment: finalResult.risk_assessment,
            urgent_actions: finalResult.urgent_actions,
            expert_reasoning: finalResult.expert_second_opinion?.medical_reasoning
          } : {
            triggered: false,
            reason: "No dangerous disagreement detected"
          },

          // 🎯 FINAL DECISION LOGIC
          final_decision: {
            diagnosis: finalResult.final_expert_diagnosis ||
              finalResult.consensus_diagnosis ||
              onnxResult.finalLabel,
            confidence: finalResult.professor_confidence ||
              finalResult.consensus_confidence ||
              (onnxResult.confidence || 0),
            decision_maker: finalResult.system_summary.final_decision_maker,
            reasoning: getFinalDecisionReasoning(finalResult)
          },

          // Cost & Performance Tracking
          performance_metrics: {
            total_processing_time: Date.now() - new Date(finalResult.analysis_metadata.timestamp).getTime(),
            gpt4o_cost_usd: finalResult.cost_tracking.gpt4o_cost_usd || 0,
            professor_cost_usd: finalResult.cost_tracking.professor_cost_usd || 0,
            total_cost_usd: finalResult.cost_tracking.total_cost_usd || 0,
            tokens_used: finalResult.cost_tracking.tokens_used || 0
          },

          // Clinical Recommendations (Enhanced)
          clinical_recommendations: finalResult.clinical_recommendations,

          // AI Synthesis
          ai_synthesis: finalResult.ai_synthesis?.success ? finalResult.ai_synthesis.synthesis : null
        }
      }
    };

    res.json(standardizedResponse);

  } catch (err) {
    console.error("❌ Lỗi xử lý phân tích ảnh:", err);
    res.status(500).json({
      error: "Đã xảy ra lỗi khi phân tích ảnh X-ray!",
      details: err.message
    });
  }
}

/**
 * API chỉ chạy GPT-4o analysis (cho testing và so sánh)
 * @param {object} req - Request object
 * @param {object} res - Response object
 */
export async function analyzeXrayGPT4oOnly(req, res) {
  try {
    const imagePath = req.file?.path;
    if (!imagePath) {
      return res.status(400).json({ error: "Không tìm thấy file ảnh!" });
    }

    if (!process.env.OPENAI_API_KEY) {
      return res.status(400).json({
        error: "OPENAI_API_KEY không được cấu hình!"
      });
    }

    const imageUrl = req.file?.cloudinaryUrl || imagePath;

    let clinical_info = {};
    if (req.body.clinical_info) {
      try {
        clinical_info = JSON.parse(req.body.clinical_info);
      } catch (err) {
        return res.status(400).json({ error: "clinical_info phải là JSON hợp lệ!" });
      }
    }

    console.log('🧠 GPT-4o only analysis started...');

    const gpt4oResult = await analyzeXrayWithGPT4o(
      imageUrl,
      clinical_info,
      'pediatric_xray'
    );

    // Standardized response format for GPT-4o only
    res.json({
      success: true,
      stage: "gpt4o_analysis_completed",
      message: "GPT-4o analysis completed successfully",
      data: {
        analysis_type: 'gpt4o_only',
        modelName: 'GPT-4o-Vision',
        timestamp: new Date().toISOString(),

        // Main GPT-4o results
        gpt4o_analysis: gpt4oResult.analysis,

        // Performance metrics
        performance_metrics: {
          cost_usd: gpt4oResult.analysis?.analysis_metadata?.cost_estimate_usd || 0,
          tokens_used: gpt4oResult.usage?.total_tokens || 0,
          response_time_ms: gpt4oResult.analysis?.analysis_metadata?.response_time_ms || 0
        },

        // Raw response (for debugging)
        raw_response: gpt4oResult
      }
    });

  } catch (err) {
    console.error("❌ Lỗi GPT-4o analysis:", err);
    res.status(500).json({
      error: "Đã xảy ra lỗi khi phân tích ảnh với GPT-4o!",
      details: err.message
    });
  }
}

/**
 * Health check API cho OpenAI service
 * @param {object} req - Request object
 * @param {object} res - Response object
 */
export async function gpt4oHealthCheck(req, res) {
  try {
    const healthStatus = await healthCheck();

    res.json({
      service: 'GPT-4o Medical Analysis',
      ...healthStatus,
      recommendations: healthStatus.status === 'healthy'
        ? ['Service is ready for medical analysis']
        : ['Check OpenAI API key', 'Verify network connectivity', 'Check API quotas']
    });

  } catch (err) {
    res.status(500).json({
      service: 'GPT-4o Medical Analysis',
      status: 'error',
      error: err.message
    });
  }
}
