import path from "path";
import { v2 as cloudinary } from "cloudinary";
import multer from "multer";
import { fileURLToPath } from "url";
import { uploadsDir } from "../index.js";

const __filename = fileURLToPath(import.meta.url);

/**
 * Middleware xử lý file DICOM và upload lên Cloudinary
 * @param {object} req - Request object
 * @param {object} res - Response object  
 * @param {function} next - Next middleware function
 */
export const handleDicomFile = async (req, res, next) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    const file = req.file;
    const isDicom = file.originalname.toLowerCase().endsWith('.dcm') ||
      file.mimetype === 'application/dicom';

    if (isDicom) {
      console.log('📄 Processing DICOM file...');
    }

    // Upload to Cloudinary
    console.log('☁️ Uploading to Cloudinary...');
    const cloudinaryResult = await cloudinary.uploader.upload(file.path, {
      folder: 'xray-diagnosis',
      resource_type: 'image',
      transformation: [
        { width: 1000, height: 1000, crop: 'limit' },
        { quality: 'auto' },
        { format: 'jpg' }
      ]
    });

    // Add Cloudinary info to req.file
    req.file.cloudinaryId = cloudinaryResult.public_id;
    req.file.cloudinaryUrl = cloudinaryResult.secure_url;

    console.log(`✅ Uploaded to Cloudinary: ${cloudinaryResult.public_id}`);
    next();

  } catch (error) {
    console.error('❌ DICOM/Cloudinary upload error:', error);
    return res.status(500).json({
      error: "File upload failed",
      details: error.message
    });
  }
};

/**
 * Middleware kiểm tra OpenAI API key (cho 3-AI system)  
 * @param {object} req - Request object
 * @param {object} res - Response object
 * @param {function} next - Next middleware function
 */
export const validateOpenAIKey = async (req, res, next) => {
  try {
    const openaiKey = process.env.OPENAI_API_KEY;
    if (!openaiKey) {
      return res.status(500).json({
        error: "OpenAI API key not configured for 3-AI system"
      });
    }

    // Simple validation - check if key has proper format
    if (!openaiKey.startsWith('sk-') || openaiKey.length < 40) {
      return res.status(500).json({
        error: "Invalid OpenAI API key format for 3-AI system"
      });
    }

    next();
  } catch (error) {
    return res.status(500).json({
      error: "OpenAI validation failed",
      details: error.message
    });
  }
};

/**
 * Middleware validation cho clinical_info
 * @param {object} req - Request object
 * @param {object} res - Response object
 * @param {function} next - Next middleware function
 */
export const validateClinicalInfo = async (req, res, next) => {
  try {
    if (req.body.clinical_info) {
      let clinical_info;

      try {
        clinical_info = JSON.parse(req.body.clinical_info);
      } catch (parseError) {
        return res.status(400).json({
          error: "clinical_info must be valid JSON"
        });
      }

      // Validate structure
      const validFields = ['age', 'symptoms', 'initial_diagnosis', 'history'];
      const invalidFields = Object.keys(clinical_info).filter(
        field => !validFields.includes(field)
      );

      if (invalidFields.length > 0) {
        return res.status(400).json({
          error: `Invalid clinical info fields: ${invalidFields.join(', ')}. Valid fields: ${validFields.join(', ')}`
        });
      }

      // Pre-parse for controller
      req.body.parsed_clinical_info = clinical_info;
    }

    next();
  } catch (error) {
    return res.status(500).json({
      error: "Clinical info validation failed",
      details: error.message
    });
  }
};

// Multer configuration for file upload
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadsDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

export const upload = multer({
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    // Accept images and DICOM files
    const allowedTypes = /jpeg|jpg|png|dcm|dicom/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype) || file.mimetype === 'application/dicom';

    if (mimetype && extname) {
      return cb(null, true);
    } else {
      cb(new Error('Only JPEG, PNG, and DICOM files are allowed!'));
    }
  }
});

export default {
  handleDicomFile,
  validateOpenAIKey,
  validateClinicalInfo,
  upload
}; 