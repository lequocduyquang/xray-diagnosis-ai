import { healthCheck } from "../services/gpt.js";

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
