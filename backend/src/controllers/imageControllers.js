import { analyzeXrayImage } from "../services/onnxService.js";
import { 
  analyzeXrayWithGPT4o, 
  compareAndSynthesizeResults, 
  getSecondOpinion,
  healthCheck 
} from "../services/gpt.js";

// Danh sách nhãn hợp lệ để kiểm tra clinical_info
const validLabels = [
  "Normal",
  "Pneumonia",
  "Bronchitis",
  "Brocho-pneumonia",
  "Other disease",
  "Bronchiolitis",
];

/**
 * API xử lý ảnh X-ray và trả kết quả phân tích
 * UPDATED: Tích hợp ONNX models + GPT-4o (tùy chọn)
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
      !validLabels.includes(clinical_info.initial_diagnosis)
    ) {
      return res.status(400).json({
        error: `Chẩn đoán lâm sàng không hợp lệ! Phải thuộc: ${validLabels.join(
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

    // Kiểm tra có enable GPT-4o không
    const enableGPT4o = req.query.enable_gpt4o === 'true' || 
                        req.body.enable_gpt4o === true ||
                        req.body.enable_gpt4o === 'true';
    
    console.log(`🤖 GPT-4o enabled: ${enableGPT4o}`);

    // 1. Luôn chạy ONNX models trước (ResNet50 + DenseNet121)
    console.log('🔬 Running ONNX models analysis...');
    const onnxResult = await analyzeXrayImage(
      imagePath,
      clinical_info,
      cloudinaryId
    );

    // Chuẩn bị response cơ bản
    let finalResult = {
      // Kết quả ONNX (giữ nguyên format cũ cho backward compatibility)
      ...onnxResult,
      
      // Thêm metadata về phân tích
      analysis_metadata: {
        timestamp: new Date().toISOString(),
        models_used: ['ResNet50-v1', 'ResNet50-v2', 'DenseNet121'],
        gpt4o_enabled: enableGPT4o,
        analysis_type: enableGPT4o ? 'enhanced' : 'standard'
      }
    };

    // 2. Nếu enable GPT-4o, chạy thêm GPT-4o analysis
    if (enableGPT4o) {
      try {
        // Kiểm tra OpenAI API key
        if (!process.env.OPENAI_API_KEY) {
          return res.status(400).json({ 
            error: "OPENAI_API_KEY không được cấu hình!" 
          });
        }

        console.log('🧠 Running GPT-4o analysis...');
        const gpt4oResult = await analyzeXrayWithGPT4o(
          imageUrl, 
          clinical_info, 
          'pediatric_xray'
        );

        // Thêm kết quả GPT-4o vào response
        finalResult.gpt4o_analysis = gpt4oResult;
        finalResult.analysis_metadata.models_used.push('GPT-4o');

        // 3. Nếu cả 2 AI đều thành công, kiểm tra disagreement và tổng hợp kết quả
        if (gpt4oResult.success && onnxResult) {
          
          // 🚨 CRITICAL: Check for dangerous disagreement
          const onnxDiagnosis = onnxResult.finalLabel;
          const gpt4oDiagnosis = gpt4oResult.analysis?.diagnosis;
          const isDangerous = checkDangerousDisagreement(onnxDiagnosis, gpt4oDiagnosis);
          
          if (isDangerous) {
            console.log("🚨 DANGEROUS DISAGREEMENT DETECTED!");
            console.log(`📊 ONNX: ${onnxDiagnosis} vs GPT-4o: ${gpt4oDiagnosis}`);
            
            // Trigger automatic second opinion
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
                finalResult.analysis_metadata.disagreement_detected = true;
                
                // Override final recommendation with expert opinion
                finalResult.final_expert_diagnosis = secondOpinionResult.expert_opinion.final_expert_diagnosis;
                finalResult.risk_assessment = secondOpinionResult.expert_opinion.risk_assessment;
                finalResult.urgent_actions = secondOpinionResult.expert_opinion.urgent_actions;
                
                console.log(`🩺 Expert final diagnosis: ${secondOpinionResult.expert_opinion.final_expert_diagnosis}`);
              } else {
                console.error("❌ Expert second opinion failed:", secondOpinionResult.error);
                finalResult.expert_second_opinion = {
                  error: "Expert consultation failed - MANUAL REVIEW REQUIRED",
                  fallback_diagnosis: "IMMEDIATE RADIOLOGIST REVIEW NEEDED"
                };
              }
            } catch (expertError) {
              console.error("❌ Second opinion error:", expertError);
              finalResult.expert_second_opinion = {
                error: "Technical error in expert consultation",
                fallback_diagnosis: "URGENT MANUAL REVIEW REQUIRED"
              };
            }
          }

          console.log('🔄 Synthesizing AI results...');
          const synthesis = await compareAndSynthesizeResults(
            gpt4oResult,
            onnxResult,
            clinical_info
          );
          
          finalResult.ai_synthesis = synthesis;
          finalResult.analysis_metadata.synthesis_available = synthesis.success;

          // Thêm so sánh confidence
          const onnxConfidence = onnxResult.confidence || 0;
          const gpt4oConfidence = gpt4oResult.analysis?.confidence || 0;
          
          finalResult.confidence_comparison = {
            onnx_models: onnxConfidence,
            gpt4o: gpt4oConfidence,
            difference: Math.abs(onnxConfidence - gpt4oConfidence),
            agreement_level: getAgreementLevel(onnxConfidence, gpt4oConfidence)
          };

          // Thêm chi phí tracking
          finalResult.cost_tracking = {
            gpt4o_cost_usd: gpt4oResult.analysis?.analysis_metadata?.cost_estimate_usd || 0,
            tokens_used: gpt4oResult.usage?.total_tokens || 0
          };

        } else if (!gpt4oResult.success) {
          finalResult.gpt4o_error = gpt4oResult.error;
        }

      } catch (gpt4oError) {
        console.error('❌ GPT-4o analysis failed:', gpt4oError);
        finalResult.gpt4o_error = gpt4oError.message;
      }
    }

    // 4. Thêm clinical recommendations
    finalResult.clinical_recommendations = generateClinicalRecommendations(
      finalResult, 
      clinical_info
    );

    console.log('✅ X-ray analysis completed');
    res.json(finalResult);

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

    res.json({
      gpt4o_analysis: gpt4oResult,
      analysis_type: 'gpt4o_only',
      timestamp: new Date().toISOString()
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

/**
 * Kiểm tra disagreement nguy hiểm giữa ONNX và GPT-4o
 * @param {string} onnxDiagnosis - Chẩn đoán từ ONNX models
 * @param {string} gpt4oDiagnosis - Chẩn đoán từ GPT-4o  
 * @returns {boolean} True nếu có disagreement nguy hiểm
 */
function checkDangerousDisagreement(onnxDiagnosis, gpt4oDiagnosis) {
  if (!onnxDiagnosis || !gpt4oDiagnosis) return false;
  
  // Normalize diagnoses to handle case differences
  const onnx = onnxDiagnosis.toLowerCase().trim();
  const gpt4o = gpt4oDiagnosis.toLowerCase().trim();
  
  // Same diagnosis = no disagreement
  if (onnx === gpt4o) return false;
  
  // Define dangerous disagreement patterns
  const dangerousPatterns = [
    // Case 1: One says Normal, other says any disease
    (onnx === 'normal' && gpt4o !== 'normal'),
    (gpt4o === 'normal' && onnx !== 'normal'),
    
    // Case 2: One says Pneumonia, other says Normal (CRITICAL!)
    (onnx === 'pneumonia' && gpt4o === 'normal'),
    (gpt4o === 'pneumonia' && onnx === 'normal'),
    
    // Case 3: Pneumonia vs other diseases (less critical but still important)
    (onnx === 'pneumonia' && !['pneumonia', 'brocho-pneumonia'].includes(gpt4o)),
    (gpt4o === 'pneumonia' && !['pneumonia', 'brocho-pneumonia'].includes(onnx)),
    
    // Case 4: Severe vs mild conditions
    (onnx === 'brocho-pneumonia' && gpt4o === 'normal'),
    (gpt4o === 'brocho-pneumonia' && onnx === 'normal')
  ];
  
  const isDangerous = dangerousPatterns.some(pattern => pattern);
  
  if (isDangerous) {
    console.log(`🚨 DANGEROUS DISAGREEMENT: ONNX(${onnxDiagnosis}) vs GPT-4o(${gpt4oDiagnosis})`);
  }
  
  return isDangerous;
}

/**
 * Xác định mức độ đồng thuận giữa 2 AI models
 * @param {number} confidence1 - Confidence của model 1
 * @param {number} confidence2 - Confidence của model 2
 * @returns {string} Mức độ đồng thuận (high/medium/low)
 */
function getAgreementLevel(confidence1, confidence2) {
  const difference = Math.abs(confidence1 - confidence2);
  
  if (difference <= 0.1) return 'high';      // Chênh lệch <= 10%
  if (difference <= 0.3) return 'medium';    // Chênh lệch <= 30%
  return 'low';                              // Chênh lệch > 30%
}

/**
 * Tạo clinical recommendations dựa trên kết quả phân tích
 * @param {object} analysisResult - Kết quả phân tích tổng hợp
 * @param {object} clinical_info - Thông tin lâm sàng
 * @returns {object} Clinical recommendations
 */
function generateClinicalRecommendations(analysisResult, clinical_info) {
  const recommendations = {
    priority: 'routine',
    actions: [],
    follow_up: [],
    warnings: []
  };

  // Dựa trên kết quả ONNX
  if (analysisResult.finalLabel === 'Pneumonia') {
    recommendations.priority = 'urgent';
    recommendations.actions.push('Xác nhận chẩn đoán với bác sĩ lâm sàng');
    recommendations.actions.push('Xét nghiệm máu (WBC, CRP)');
    recommendations.actions.push('Cân nhắc điều trị kháng sinh');
  } else if (analysisResult.finalLabel === 'Normal') {
    recommendations.actions.push('Theo dõi triệu chứng lâm sàng');
    recommendations.follow_up.push('Tái khám nếu có triệu chứng mới');
  }

  // Dựa trên kết quả GPT-4o
  if (analysisResult.gpt4o_analysis?.success && analysisResult.gpt4o_analysis.analysis?.recommendations) {
    recommendations.actions.push(...analysisResult.gpt4o_analysis.analysis.recommendations);
  }

  // Dựa trên synthesis
  if (analysisResult.ai_synthesis?.success) {
    const synthesis = analysisResult.ai_synthesis.synthesis;
    if (synthesis.final_recommendation) {
      recommendations.actions.push(synthesis.final_recommendation);
    }
  }

  // Warning về agreement level
  if (analysisResult.confidence_comparison?.agreement_level === 'low') {
    recommendations.warnings.push('Độ tin cậy giữa các AI models thấp - cần xem xét kỹ');
    recommendations.priority = 'urgent';
  }

  // Warning về clinical correlation
  if (clinical_info.initial_diagnosis && 
      analysisResult.finalLabel !== clinical_info.initial_diagnosis) {
    recommendations.warnings.push(
      `Mâu thuẫn giữa AI (${analysisResult.finalLabel}) và lâm sàng (${clinical_info.initial_diagnosis})`
    );
  }

  return recommendations;
}
