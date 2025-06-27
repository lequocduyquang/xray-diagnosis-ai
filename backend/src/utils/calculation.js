/**
 * Chuyển đổi logits thành xác suất bằng softmax
 * @param {Object} logits Logits từ mô hình ONNX
 * @returns {Array<number>} Xác suất cho từng lớp
 */
export function softmax(logits) {
  const expValues = Object.values(logits).map((x) => Math.exp(x));
  const sumExp = expValues.reduce((a, b) => a + b, 0);
  return expValues.map((x) => x / sumExp);
}

/**
 * Lấy lớp dự đoán từ xác suất
 * @param {Array<number>} probabilities Xác suất cho từng lớp
 * @returns {number} Lớp dự đoán (chỉ số của lớp có xác suất cao nhất)
 */
export function getPredictedClass(probabilities) {
  return probabilities.indexOf(Math.max(...probabilities));
}

export function sigmoid(x) {
  return x.map(v => 1 / (1 + Math.exp(-v)));
}

/**
 * Xác định mức độ đồng thuận giữa 2 AI models
 * @param {number} confidence1 - Confidence của model 1
 * @param {number} confidence2 - Confidence của model 2
 * @returns {string} Mức độ đồng thuận (high/medium/low)
 */
export function getAgreementLevel(confidence1, confidence2) {
  const confidenceDiff = Math.abs(confidence1 - confidence2);

  if (confidenceDiff < 0.2) return "high";
  if (confidenceDiff < 0.5) return "medium";
  return "low";
}

/**
 * Kiểm tra disagreement nguy hiểm giữa ONNX và GPT-4o
 * ENHANCED: Covers all 6 valid labels with medical severity classification
 * Valid Labels: Normal, Pneumonia, Bronchitis, Brocho-pneumonia, Other disease, Bronchiolitis
 * @param {string} onnxDiagnosis - Chẩn đoán từ ONNX models
 * @param {string} gpt4oDiagnosis - Chẩn đoán từ GPT-4o  
 * @returns {boolean} True nếu có disagreement nguy hiểm
 */
export function checkDangerousDisagreement(onnxDiagnosis, gpt4oDiagnosis) {
  if (!onnxDiagnosis || !gpt4oDiagnosis) return false;

  // Normalize diagnoses to handle case differences
  const onnx = onnxDiagnosis.toLowerCase().trim();
  const gpt4o = gpt4oDiagnosis.toLowerCase().trim();

  // Same diagnosis = no disagreement
  if (onnx === gpt4o) return false;

  // 🏥 MEDICAL SEVERITY CLASSIFICATION
  const medicalSeverity = {
    // Level 0: No disease
    'normal': { level: 0, urgent: false, category: 'healthy' },

    // Level 1: Mild respiratory conditions  
    'bronchitis': { level: 1, urgent: false, category: 'mild' },
    'bronchiolitis': { level: 1, urgent: false, category: 'mild' },

    // Level 2: Moderate/unknown conditions
    'other disease': { level: 2, urgent: true, category: 'moderate' },

    // Level 3: Severe respiratory conditions (CRITICAL)
    'pneumonia': { level: 3, urgent: true, category: 'severe' },
    'brocho-pneumonia': { level: 3, urgent: true, category: 'severe' }
  };

  const onnxSeverity = medicalSeverity[onnx] || { level: 2, urgent: true, category: 'unknown' };
  const gpt4oSeverity = medicalSeverity[gpt4o] || { level: 2, urgent: true, category: 'unknown' };

  // 🚨 DANGEROUS DISAGREEMENT PATTERNS
  const dangerousConditions = [

    // 🔴 CRITICAL PATTERN 1: Normal vs Any Disease (FALSE NEGATIVE risk)
    (onnxSeverity.level === 0 && gpt4oSeverity.level > 0),
    (gpt4oSeverity.level === 0 && onnxSeverity.level > 0),

    // 🔴 CRITICAL PATTERN 2: Severe Disease vs Normal (LIFE THREATENING!)
    (onnxSeverity.level >= 3 && gpt4oSeverity.level === 0),
    (gpt4oSeverity.level >= 3 && onnxSeverity.level === 0),

    // 🟡 HIGH PRIORITY PATTERN 3: Severe vs Mild (significant treatment difference)
    (onnxSeverity.level >= 3 && gpt4oSeverity.level <= 1),
    (gpt4oSeverity.level >= 3 && onnxSeverity.level <= 1),

    // 🟡 HIGH PRIORITY PATTERN 4: Pneumonia family vs Other conditions
    (['pneumonia', 'brocho-pneumonia'].includes(onnx) && !['pneumonia', 'brocho-pneumonia', 'bronchitis', 'bronchiolitis'].includes(gpt4o)),
    (['pneumonia', 'brocho-pneumonia'].includes(gpt4o) && !['pneumonia', 'brocho-pneumonia', 'bronchitis', 'bronchiolitis'].includes(onnx)),

    // 🟡 IMPORTANT PATTERN 5: Bronchitis/Bronchiolitis vs Pneumonia (treatment differs)
    (['bronchitis', 'bronchiolitis'].includes(onnx) && ['pneumonia', 'brocho-pneumonia'].includes(gpt4o)),
    (['bronchitis', 'bronchiolitis'].includes(gpt4o) && ['pneumonia', 'brocho-pneumonia'].includes(onnx)),

    // 🟡 IMPORTANT PATTERN 6: Other disease disagreements (could mask serious conditions)
    (onnx === 'other disease' && gpt4o === 'normal'),
    (gpt4o === 'other disease' && onnx === 'normal'),
    (onnx === 'other disease' && ['pneumonia', 'brocho-pneumonia'].includes(gpt4o)),
    (gpt4o === 'other disease' && ['pneumonia', 'brocho-pneumonia'].includes(onnx)),

    // 🟡 IMPORTANT PATTERN 7: Major severity level gap (≥2 levels)
    (Math.abs(onnxSeverity.level - gpt4oSeverity.level) >= 2),

    // 🟡 IMPORTANT PATTERN 8: Cross-category disagreement (different medical approaches)
    (onnxSeverity.category !== gpt4oSeverity.category &&
      onnxSeverity.category !== 'unknown' &&
      gpt4oSeverity.category !== 'unknown')
  ];

  const isDangerous = dangerousConditions.some(condition => condition);

  if (isDangerous) {
    const severityGap = Math.abs(onnxSeverity.level - gpt4oSeverity.level);
    const riskLevel = severityGap >= 3 ? 'CRITICAL' :
      severityGap >= 2 ? 'HIGH' :
        onnxSeverity.urgent || gpt4oSeverity.urgent ? 'MODERATE' : 'LOW';

    console.log(`🚨 DANGEROUS DISAGREEMENT DETECTED:`);
    console.log(`   📊 ONNX: ${onnxDiagnosis} (Level ${onnxSeverity.level}, ${onnxSeverity.category})`);
    console.log(`   🧠 GPT-4o: ${gpt4oDiagnosis} (Level ${gpt4oSeverity.level}, ${gpt4oSeverity.category})`);
    console.log(`   ⚡ Severity Gap: ${severityGap} levels`);
    console.log(`   🚨 Risk Level: ${riskLevel}`);
    console.log(`   🏥 Medical Impact: ${onnxSeverity.urgent || gpt4oSeverity.urgent ? 'Urgent care needed' : 'Standard care'}`);
  }

  return isDangerous;
}


/**
 * Tạo reasoning cho final decision của 3-AI system
 * @param {object} finalResult - Kết quả phân tích cuối cùng
 * @returns {string} Reasoning text
 */
export function getFinalDecisionReasoning(finalResult) {
  const meta = finalResult.analysis_metadata;

  if (meta.professor_ai_used) {
    return `Professor AI resolved dangerous disagreement between ONNX Models (${meta.onnx_diagnosis}) and GPT-4o (${meta.gpt4o_diagnosis}). Expert medical analysis was required for patient safety.`;
  }

  if (meta.ai_agreement) {
    const source = finalResult.consensus_source || "AI Models";
    return `All AI models reached consensus. Final diagnosis based on ${source} with highest confidence score.`;
  }

  if (meta.disagreement_detected) {
    return `Disagreement detected but not dangerous. Using model with higher confidence score.`;
  }

  return `Standard 3-AI analysis completed. Diagnosis based on combined AI assessment.`;
}

/**
 * Tạo clinical recommendations dựa trên kết quả phân tích
 * @param {object} analysisResult - Kết quả phân tích tổng hợp
 * @param {object} clinical_info - Thông tin lâm sàng
 * @returns {object} Clinical recommendations
 */
export function generateClinicalRecommendations(analysisResult, clinical_info) {
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
