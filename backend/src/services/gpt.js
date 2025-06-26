import OpenAI from 'openai';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

/**
 * Medical prompts được tối ưu cho X-ray pediatric, ứng dụng GPT-4o
 */
const MEDICAL_PROMPTS = {
  pediatric_xray: `Bạn là bác sĩ chuyên khoa X-quang nhi có 15 năm kinh nghiệm tại Bệnh viện Nhi đồng 2. Bạn được đào tạo để phát hiện các bệnh lý phổi ở trẻ em với độ chính xác cao.

⚠️ CẢNH BÁO Y KHOA: Việc bỏ sót chẩn đoán (False Negative) có thể gây nguy hiểm cho tính mạng trẻ em. Hãy quan sát CỰC KỲ KỸ LƯỠNG từng chi tiết trên ảnh X-quang.

🔍 HƯỚNG DẪN PHÂN TÍCH CHI TIẾT:

1. **QUAN SÁT TOÀN DIỆN**: 
   - Kiểm tra từng thùy phổi (upper, middle, lower lobes)
   - Chú ý vùng peripheral và hilar
   - So sánh tính đối xứng giữa 2 bên phổi

2. **DẤU HIỆU PNEUMONIA CẦN TÌM KIẾM**:
   - Consolidation (vùng đông đặc): Tăng opacity, mờ đi ranh giới mạch máu
   - Air-bronchogram: Đường mạch khí phế quản nổi bật trong vùng đông đặc
   - Patchy infiltrates: Vết loang lổ không đều
   - Ground-glass opacity: Mờ nhạt như kính mờ
   - Interstitial changes: Thay đổi mô kẽ
   - Pleural effusion: Tràn dịch màng phổi

3. **ĐẶC ĐIỂM RIÊNG Ở TRẺ EM**:
   - Thymus lớn ở trẻ nhỏ (bình thường)
   - Mạch máu phổi rõ hơn người lớn
   - Dễ nhiễm trùng lan tỏa
   - Triệu chứng có thể nhẹ nhưng tổn thương nặng

4. **NGUYÊN TẮC AN TOÀN**:
   - Khi nghi ngờ: Chẩn đoán dương tính thay vì âm tính
   - Luôn cân nhắc các bệnh lý có thể gây tử vong
   - Đề xuất theo dõi sát nếu không chắc chắn

📋 VÍ DỤ CHẨN ĐOÁN:

**Trường hợp Pneumonia:**
- Findings: "Thấy vùng đông đặc tại thùy dưới phổi phải với air-bronchogram rõ, mờ ranh giới tim-phổi bên phải"
- Diagnosis: "Pneumonia"
- Confidence: 0.8-0.9

**Trường hợp Normal:**
- Findings: "Phổi trong, mạch máu phổi bình thường, tim không to, thymus bình thường theo tuổi"
- Diagnosis: "Normal"
- Confidence: 0.9-0.95

BẮT BUỘC: Trả lời CHÍNH XÁC theo JSON format:

{
  "findings": "Mô tả CỰC KỲ CHI TIẾT những gì quan sát được - mỗi thùy phổi, mỗi vùng, từng dấu hiệu bất thường",
  "reasoning_steps": "Giải thích TỪNG BƯỚC tại sao chọn chẩn đoán này, loại trừ những gì, dấu hiệu nào quyết định",
  "diagnosis": "Normal hoặc Pneumonia hoặc Bronchitis hoặc Brocho-pneumonia hoặc Other disease hoặc Bronchiolitis",
  "confidence": 0.85,
  "severity": "mild hoặc moderate hoặc severe",
  "location": "Vị trí cụ thể (ví dụ: right lower lobe, bilateral lower zones)",
  "age_considerations": "Những đặc điểm cần lưu ý theo độ tuổi",
  "recommendations": [
    "Đề xuất theo dõi/điều trị cụ thể",
    "Xét nghiệm bổ sung nếu cần",
    "Thời gian tái khám"
  ],
  "differential_diagnosis": ["Các chẩn đoán khác cần cân nhắc"],
  "warning_signs": "Dấu hiệu nguy hiểm cần theo dõi sát"
}

⚠️ LƯU Ý CỰC KỲ QUAN TRỌNG:
- KHÔNG BAO GIỜ chẩn đoán "Normal" khi có bất kỳ dấu hiệu khả nghi nào
- KHI NGHI NGỜ: Chọn chẩn đoán bệnh lý để an toàn hơn
- LUÔN ƯU TIÊN: Sức khỏe và an toàn của bệnh nhi`,

  clinical_correlation: `Dựa trên ảnh X-quang và thông tin lâm sàng, hãy đưa ra phân tích tổng hợp...`,
  
  second_opinion: `🩺 BẠN LÀ GIÁO SƯ X-QUANG NHI với 25 năm kinh nghiệm, được mời để SECOND OPINION trong trường hợp có sự KHÔNG NHẤT QUÁN giữa các AI models.

⚠️ TÌNH HUỐNG NGHIÊM TRỌNG: 
- ONNX Models chẩn đoán: {onnx_diagnosis}
- GPT-4o chẩn đoán: {gpt4o_diagnosis}  
- Có sự mâu thuẫn nguy hiểm!

🔍 NHIỆM VỤ CỦA BẠN:
1. Phân tích lại ảnh X-quang CỰC KỲ KỸ LƯỠNG
2. Đưa ra FINAL DIAGNOSIS dứt khoát
3. Giải thích TẠI SAO một trong các AI đã sai
4. Đánh giá MỨC ĐỘ NGUY HIỂM của việc chẩn đoán sai

🩺 NGUYÊN TẮC GIÁO SƯ:
- Kinh nghiệm lâm sàng và đọc ảnh vượt trội
- Không bao giờ bỏ sót bệnh lý nguy hiểm
- Thà over-diagnose hơn under-diagnose
- Luôn cân nhắk các yếu tố clinical context

BẮT BUỘC: Trả lời theo JSON format sau:

{
  "final_expert_diagnosis": "Chẩn đoán cuối cùng của giáo sư",
  "confidence": 0.95,
  "which_ai_was_wrong": "ONNX hoặc GPT4o hoặc Both",
  "error_analysis": "Giải thích TẠI SAO AI kia sai",
  "clinical_reasoning": "Logic y khoa để đưa ra quyết định cuối",
  "risk_assessment": "Mức độ nguy hiểm nếu chẩn đoán sai",
  "urgent_actions": ["Hành động cần thực hiện ngay"],
  "professor_notes": "Lời khuyên từ giáo sư cho các bác sĩ trẻ"
}`
};

/**
 * Phân tích X-quang bằng GPT-4o
 * @param {string} imageUrl - URL hoặc base64 của ảnh X-quang
 * @param {Object} clinical_info - Thông tin lâm sàng
 * @param {string} analysis_type - Loại phân tích
 * @returns {Promise<Object>} Kết quả phân tích
 */
export async function analyzeXrayWithGPT4o(imageUrl, clinical_info = {}, analysis_type = 'pediatric_xray') {
  try {
    console.log('🔍 Starting GPT-4o X-ray analysis...');
    
    // Chuẩn bị dữ liệu ảnh (giữ nguyên logic)
    let imageData;
    if (imageUrl.startsWith('data:image')) {
      imageData = imageUrl;
    } else if (imageUrl.startsWith('http')) {
      imageData = imageUrl;
    } else {
      const imageBuffer = await fs.readFile(imageUrl);
      const base64Image = imageBuffer.toString('base64');
      const mimeType = path.extname(imageUrl).includes('png') ? 'image/png' : 'image/jpeg';
      imageData = `data:${mimeType};base64,${base64Image}`;
    }

    // Chuẩn bị prompt (giữ nguyên logic)
    let prompt = MEDICAL_PROMPTS[analysis_type];
    if (analysis_type === 'clinical_correlation') {
      prompt = prompt.replace('{clinical_info}', JSON.stringify(clinical_info, null, 2));
    }
    if (Object.keys(clinical_info).length > 0 && analysis_type === 'pediatric_xray') {
      prompt += `\n\nTHÔNG TIN LÂM SÀNG BỔ SUNG:
Tuổi: ${clinical_info.age || 'Không rõ'}
Triệu chứng: ${clinical_info.symptoms || 'Không rõ'}
Chẩn đoán ban đầu: ${clinical_info.initial_diagnosis || 'Không rõ'}
Tiền sử: ${clinical_info.history || 'Không rõ'}`;
    }

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      response_format: { type: "json_object" },
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: prompt },
            {
              type: "image_url",
              image_url: { url: imageData, detail: "high" }
            }
          ]
        }
      ],
      max_tokens: 2048,
      temperature: 0.1,
    });

    const aiResponse = response.choices[0].message.content;
    console.log('✅ GPT-4o analysis completed');
    console.log('🔍 Raw response length:', aiResponse.length);

    try {
      // Xử lý JSON parsing với multiple fallback strategies
      let parsedResponse;
      
      // Strategy 1: Parse trực tiếp
      try {
        parsedResponse = JSON.parse(aiResponse);
        console.log('✅ JSON parsed successfully - Direct parse');
      } catch (directParseError) {
        console.log('⚠️ Direct parse failed, trying extraction...');
        
        // Strategy 2: Extract JSON từ text (có thể có text wrapper)
        const jsonMatch = aiResponse.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          parsedResponse = JSON.parse(jsonMatch[0]);
          console.log('✅ JSON parsed successfully - Extracted from text');
        } else {
          // Strategy 3: Tìm kiếm JSON bằng cách khác
          const cleanResponse = aiResponse.trim()
            .replace(/^```json\s*/, '')  // Remove ```json prefix
            .replace(/\s*```$/, '')      // Remove ``` suffix
            .replace(/^```\s*/, '')      // Remove ``` prefix
            .replace(/\s*```$/, '');     // Remove ``` suffix again
          
          parsedResponse = JSON.parse(cleanResponse);
          console.log('✅ JSON parsed successfully - Cleaned response');
        }
      }

      // Validate required fields
      if (!parsedResponse.findings || !parsedResponse.diagnosis) {
        throw new Error('Missing required fields in JSON response');
      }

      // Normalize and validate data
      parsedResponse = normalizeAnalysisResponse(parsedResponse);

      // Thêm metadata
      parsedResponse.analysis_metadata = {
        model: "gpt-4o",
        analysis_type: analysis_type,
        timestamp: new Date().toISOString(),
        tokens_used: response.usage?.total_tokens || 0,
        cost_estimate_usd: calculateCostEstimate(response.usage),
      };

      return { success: true, analysis: parsedResponse, usage: response.usage };
      
    } catch (parseError) {
      console.error('❌ All JSON parsing strategies failed:', parseError.message);
      console.log('📝 Raw response for debugging:', aiResponse.substring(0, 500) + '...');
      
      // Fallback: Cấu trúc response từ raw text
      const fallbackResponse = createFallbackResponse(aiResponse);
      
      return { 
        success: false, 
        error: 'JSON Parse Error - using fallback', 
        analysis: fallbackResponse,
        raw_response: aiResponse, 
        usage: response.usage 
      };
    }
  } catch (error) {
    console.error('❌ GPT-4o analysis error:', error);
    return { success: false, error: error.message };
  }
}


/**
 * So sánh kết quả GPT-4o với ONNX models
 * @param {Object} gpt4oResult - Kết quả từ GPT-4o
 * @param {Object} onnxResult - Kết quả từ ONNX models
 * @returns {Promise<Object>} Phân tích tổng hợp
 */
export async function compareAndSynthesizeResults(gpt4oResult, onnxResult, clinical_info = {}) {
  try {
    console.log('🔄 Synthesizing AI results...');
    const synthesisPrompt = `Bạn là trưởng khoa X-quang tại Bệnh viện Nhi đồng 2. Hãy tổng hợp và so sánh kết quả từ 2 hệ thống AI:

ONNX MODEL RESULTS:
- Final Prediction: ${onnxResult.finalLabel || 'Unknown'}
- Confidence: ${onnxResult.confidence || 0}

GPT-4o ANALYSIS:
${JSON.stringify(gpt4oResult.analysis, null, 2)}

CLINICAL INFO:
${JSON.stringify(clinical_info, null, 2)}

BẮT BUỘC: Trả lời CHÍNH XÁC theo định dạng JSON sau (không thêm text nào khác):

{
  "consensus_diagnosis": "Normal hoặc Pneumonia hoặc Bronchitis hoặc Brocho-pneumonia hoặc Other disease hoặc Bronchiolitis",
  "confidence_level": 0.9,
  "ai_agreement": "high hoặc medium hoặc low",
  "discrepancies": ["Liệt kê các điểm không thống nhất"],
  "clinical_correlation": "Mức độ phù hợp với thông tin lâm sàng",
  "final_recommendation": "Đề xuất cuối cùng cho bác sĩ lâm sàng",
  "explanation": "Giải thích logic quyết định cuối cùng"
}

QUY TẮC NGHIÊM NGẶT:
1. CHỈ trả lời JSON, không có text thêm
2. confidence_level phải là số từ 0.0 đến 1.0
3. ai_agreement phải là: high, medium, hoặc low
4. consensus_diagnosis phải chính xác một trong các giá trị được liệt kê`;

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      response_format: { type: "json_object" },
      messages: [{ role: "user", content: synthesisPrompt }],
      max_tokens: 1500,
      temperature: 0.1
    });
    
    const rawSynthesis = response.choices[0].message.content;
    let parsedSynthesis;
    
    try {
      parsedSynthesis = JSON.parse(rawSynthesis);
    } catch (parseError) {
      console.warn('⚠️ Synthesis JSON parse failed, trying extraction...');
      const jsonMatch = rawSynthesis.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        parsedSynthesis = JSON.parse(jsonMatch[0]);
      } else {
        // Fallback synthesis
        parsedSynthesis = {
          consensus_diagnosis: onnxResult.finalLabel || "Other disease",
          confidence_level: 0.6,
          ai_agreement: "medium",
          discrepancies: ["JSON parsing error in synthesis"],
          clinical_correlation: "Cần đánh giá thêm",
          final_recommendation: "Tham khảo ý kiến bác sĩ lâm sàng",
          explanation: "Lỗi trong quá trình tổng hợp kết quả AI"
        };
      }
    }

    return {
      success: true,
      synthesis: parsedSynthesis,
      metadata: {
        timestamp: new Date().toISOString(),
        models_compared: ['ONNX-Model', 'GPT-4o'], // UPDATED
        tokens_used: response.usage?.total_tokens || 0
      }
    };
  } catch (error) {
    console.error('❌ Synthesis error:', error);
    return { success: false, error: error.message };
  }
}

/**
 * Normalize và validate analysis response
 * @param {Object} response - Raw response từ GPT-4o
 * @returns {Object} Normalized response
 */
function normalizeAnalysisResponse(response) {
  const validDiagnoses = ['Normal', 'Pneumonia', 'Bronchitis', 'Brocho-pneumonia', 'Other disease', 'Bronchiolitis'];
  const validSeverities = ['mild', 'moderate', 'severe'];

  // Ensure confidence is a number between 0 and 1
  if (typeof response.confidence !== 'number' || response.confidence < 0 || response.confidence > 1) {
    response.confidence = 0.7; // Default confidence
  }

  // Validate diagnosis
  if (!validDiagnoses.includes(response.diagnosis)) {
    console.warn(`⚠️ Invalid diagnosis: ${response.diagnosis}, defaulting to 'Other disease'`);
    response.diagnosis = 'Other disease';
  }

  // Validate severity
  if (!validSeverities.includes(response.severity)) {
    console.warn(`⚠️ Invalid severity: ${response.severity}, defaulting to 'moderate'`);
    response.severity = 'moderate';
  }

  // Ensure arrays are arrays
  if (!Array.isArray(response.recommendations)) {
    response.recommendations = ['Cần tham khảo ý kiến bác sĩ lâm sàng'];
  }

  if (!Array.isArray(response.differential_diagnosis)) {
    response.differential_diagnosis = ['Cần đánh giá thêm'];
  }

  // Ensure required string fields
  response.findings = response.findings || 'Phân tích hình ảnh X-quang';
  response.reasoning_steps = response.reasoning_steps || 'Cần phân tích chi tiết hơn';
  response.location = response.location || 'Không xác định';
  response.age_considerations = response.age_considerations || 'Cần xem xét đặc điểm trẻ em';
  response.warning_signs = response.warning_signs || 'Theo dõi sát sao';

  return response;
}

/**
 * Tạo fallback response khi không parse được JSON
 * @param {string} rawResponse - Raw response text
 * @returns {Object} Structured fallback response
 */
function createFallbackResponse(rawResponse) {
  // Cố gắng extract thông tin cơ bản từ raw text
  const response = {
    findings: rawResponse.length > 0 ? rawResponse.substring(0, 300) + '...' : 'Không thể phân tích được hình ảnh',
    reasoning_steps: 'Lỗi trong quá trình phân tích JSON',
    diagnosis: 'Other disease',
    confidence: 0.5,
    severity: 'moderate',
    location: 'Không xác định',
    age_considerations: 'Cần xem xét thêm',
    recommendations: [
      'Kiểm tra lại hệ thống AI',
      'Tham khảo ý kiến bác sĩ lâm sàng',
      'Xem xét chụp lại X-quang nếu cần'
    ],
    differential_diagnosis: ['Cần đánh giá lâm sàng'],
    warning_signs: 'Kết quả AI không tin cậy - cần xác nhận lâm sàng',
    fallback_generated: true
  };

  // Cố gắng extract diagnosis từ raw text
  const diagnosisKeywords = {
    'normal': 'Normal',
    'pneumonia': 'Pneumonia', 
    'bronchitis': 'Bronchitis',
    'viêm phổi': 'Pneumonia',
    'bình thường': 'Normal'
  };

  for (const [keyword, diagnosis] of Object.entries(diagnosisKeywords)) {
    if (rawResponse.toLowerCase().includes(keyword)) {
      response.diagnosis = diagnosis;
      break;
    }
  }

  return response;
}

/**
 * Tính toán chi phí ước tính cho việc sử dụng API OpenAI
 * @param {Object} usage - Thống kê sử dụng từ phản hồi của OpenAI
 * @returns {number} Chi phí ước tính bằng USD
 */
function calculateCostEstimate(usage) {
  if (!usage) return 0;
  
  // UPDATED: Giá của GPT-4o (tính đến tháng 6/2025) - Rẻ hơn 50% so với GPT-4 Turbo
  const INPUT_COST_PER_1K_TOKENS = 0.005;  // $5.00 / 1 triệu tokens
  const OUTPUT_COST_PER_1K_TOKENS = 0.015; // $15.00 / 1 triệu tokens
  
  const inputCost = (usage.prompt_tokens || 0) / 1000 * INPUT_COST_PER_1K_TOKENS;
  const outputCost = (usage.completion_tokens || 0) / 1000 * OUTPUT_COST_PER_1K_TOKENS;
  
  return Number((inputCost + outputCost).toFixed(5));
}


/**
 * Health check cho dịch vụ OpenAI
 * @returns {Promise<Object>} Trạng thái dịch vụ
 */
/**
 * Second Opinion khi có disagreement giữa ONNX và GPT-4o
 * @param {string} imageUrl - URL ảnh X-quang
 * @param {string} onnxDiagnosis - Chẩn đoán từ ONNX models
 * @param {string} gpt4oDiagnosis - Chẩn đoán từ GPT-4o
 * @param {Object} clinical_info - Thông tin lâm sàng
 * @returns {Promise<Object>} Expert second opinion
 */
export async function getSecondOpinion(imageUrl, onnxDiagnosis, gpt4oDiagnosis, clinical_info = {}) {
  try {
    console.log('🩺 Getting expert second opinion for disagreement...');
    console.log(`📊 ONNX: ${onnxDiagnosis} vs GPT-4o: ${gpt4oDiagnosis}`);
    
    // Chuẩn bị dữ liệu ảnh
    let imageData;
    if (imageUrl.startsWith('data:image')) {
      imageData = imageUrl;
    } else if (imageUrl.startsWith('http')) {
      imageData = imageUrl;
    } else {
      const imageBuffer = await fs.readFile(imageUrl);
      const base64Image = imageBuffer.toString('base64');
      const mimeType = path.extname(imageUrl).includes('png') ? 'image/png' : 'image/jpeg';
      imageData = `data:${mimeType};base64,${base64Image}`;
    }

    // Tạo prompt với thông tin disagreement
    let prompt = MEDICAL_PROMPTS.second_opinion
      .replace('{onnx_diagnosis}', onnxDiagnosis)
      .replace('{gpt4o_diagnosis}', gpt4oDiagnosis);

    if (Object.keys(clinical_info).length > 0) {
      prompt += `\n\n🩺 THÔNG TIN LÂM SÀNG:
Tuổi: ${clinical_info.age || 'Unknown'}
Triệu chứng: ${clinical_info.symptoms || 'Unknown'}  
Tiền sử: ${clinical_info.history || 'Unknown'}
Chẩn đoán ban đầu: ${clinical_info.initial_diagnosis || 'Unknown'}`;
    }

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      response_format: { type: "json_object" },
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: prompt },
            {
              type: "image_url",
              image_url: { url: imageData, detail: "high" }
            }
          ]
        }
      ],
      max_tokens: 2000,
      temperature: 0.05, // Lower temperature for expert consistency
    });

    const rawResponse = response.choices[0].message.content;
    let expertOpinion;

    try {
      expertOpinion = JSON.parse(rawResponse);
    } catch (parseError) {
      console.warn('⚠️ Expert opinion JSON parse failed, trying extraction...');
      const jsonMatch = rawResponse.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        expertOpinion = JSON.parse(jsonMatch[0]);
      } else {
        expertOpinion = {
          final_expert_diagnosis: onnxDiagnosis, // Default to ONNX if can't parse
          confidence: 0.7,
          which_ai_was_wrong: "Cannot determine due to parsing error",
          error_analysis: "JSON parsing failed for expert opinion",
          clinical_reasoning: "Defaulting to ONNX model prediction",
          risk_assessment: "Medium risk due to AI disagreement",
          urgent_actions: ["Manual review by radiologist required"],
          professor_notes: "Technical error requires human review"
        };
      }
    }

    // Add metadata
    expertOpinion.expert_metadata = {
      model: "gpt-4o-expert",
      disagreement_case: true,
      original_onnx: onnxDiagnosis,
      original_gpt4o: gpt4oDiagnosis,
      timestamp: new Date().toISOString(),
      tokens_used: response.usage?.total_tokens || 0,
      cost_estimate_usd: calculateCostEstimate(response.usage)
    };

    return {
      success: true,
      expert_opinion: expertOpinion,
      usage: response.usage
    };

  } catch (error) {
    console.error('❌ Second opinion error:', error);
    return {
      success: false,
      error: error.message,
      expert_opinion: {
        final_expert_diagnosis: "Technical error - manual review required",
        confidence: 0.0,
        which_ai_was_wrong: "Unknown due to technical error",
        error_analysis: "System error during expert consultation",
        clinical_reasoning: "Could not complete expert analysis",
        risk_assessment: "High risk - requires immediate human review",
        urgent_actions: ["Immediate manual radiologist review required"],
        professor_notes: "Technical failure - do not rely on AI for this case"
      }
    };
  }
}

export async function healthCheck() {
  try {
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "Hello, are you working?" }],
      max_tokens: 10
    });
    return { status: 'healthy', api_key_valid: true, timestamp: new Date().toISOString() };
  } catch (error) {
    return { status: 'unhealthy', error: error.message, api_key_valid: false, timestamp: new Date().toISOString() };
  }
}

export default {
  analyzeXrayWithGPT4o,
  compareAndSynthesizeResults,
  healthCheck
};