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
  pediatric_xray: `Bạn là bác sĩ chuyên khoa X-quang nhi tại Bệnh viện Nhi đồng 2 với 15 năm kinh nghiệm.

NHIỆM VỤ: Phân tích ảnh X-quang phổi trẻ em và đưa ra chẩn đoán chính xác ở định dạng JSON.

ĐỊNH DẠNG PHẢN HỒI (JSON BẮT BUỘC):
{
  "findings": "Mô tả chi tiết, có cấu trúc những gì quan sát được trên ảnh.",
  "reasoning_steps": "Giải thích từng bước logic để đi đến chẩn đoán, ví dụ: 'Thấy vùng đông đặc ở thùy dưới phổi phải, kèm dấu hiệu air-bronchogram...'",
  "diagnosis": "Chẩn đoán chính (Normal/Pneumonia/Bronchitis/etc)",
  "confidence": 0.85,
  "severity": "mild/moderate/severe",
  "location": "right/left/bilateral lower lobe, etc",
  "age_considerations": "Các đặc điểm X-quang riêng biệt cần xem xét ở độ tuổi này (ví dụ: tuyến ức lớn sinh lý).",
  "recommendations": [
    "Đề xuất xét nghiệm bổ sung nếu cần (ví dụ: CT scan, xét nghiệm máu).",
    "Theo dõi lâm sàng.", 
    "Gợi ý hướng điều trị."
  ],
  "differential_diagnosis": ["Liệt kê các chẩn đoán phân biệt có khả năng nhất."],
  "warning_signs": "Các dấu hiệu cảnh báo nguy hiểm cần chú ý ngay."
}

LƯU Ý QUAN TRỌNG:
- Luôn ưu tiên an toàn của bệnh nhân.
- Xem xét kỹ các đặc điểm giải phẫu đặc thù ở trẻ em.
- Đưa ra độ tin cậy một cách thực tế.
- Luôn nhấn mạnh rằng kết quả này cần được bác sĩ lâm sàng xác nhận.`,

  clinical_correlation: `Dựa trên ảnh X-quang và thông tin lâm sàng, hãy đưa ra phân tích tổng hợp...`, // Giữ nguyên hoặc tùy chỉnh thêm
  second_opinion: `Bạn là bác sĩ X-quang senior đang review case khó...` // Giữ nguyên hoặc tùy chỉnh thêm
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

    try {
      // Vì đã bật JSON mode, có thể parse trực tiếp
      const parsedResponse = JSON.parse(aiResponse);

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
      console.warn('⚠️ Could not parse JSON response despite JSON mode, returning raw text');
      return { 
          success: false, 
          error: 'JSON Parse Error', 
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

HÃY CUNG CẤP PHÂN TÍCH TỔNG HỢP Ở ĐỊNH DẠNG JSON:
{
  "consensus_diagnosis": "Chẩn đoán thống nhất cuối cùng",
  "confidence_level": "Mức độ tin cậy của chẩn đoán thống nhất (ví dụ: 0.9)",
  "ai_agreement": "Mức độ đồng thuận giữa 2 AI (high/medium/low)",
  "discrepancies": ["Liệt kê các điểm không thống nhất, nếu có."],
  "clinical_correlation": "Mức độ phù hợp tổng thể với thông tin lâm sàng.",
  "final_recommendation": "Đề xuất cuối cùng cho bác sĩ lâm sàng.",
  "explanation": "Giải thích chi tiết logic để đưa ra quyết định cuối cùng."
}`;

    const response = await openai.chat.completions.create({
      model: "gpt-4o", // UPDATED: Sử dụng model GPT-4o
      response_format: { type: "json_object" }, // NEW: Bật chế độ JSON
      messages: [{ role: "user", content: synthesisPrompt }],
      max_tokens: 1500,
      temperature: 0.1
    });
    
    const parsedSynthesis = JSON.parse(response.choices[0].message.content);

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