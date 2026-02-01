import React, { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { ModelInfo } from "../../lib/types";

interface OnboardingProps {
  onModelSelected: () => void;
}

const Onboarding: React.FC<OnboardingProps> = ({ onModelSelected }) => {
  const [availableModels, setAvailableModels] = useState<ModelInfo[]>([]);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      const models: ModelInfo[] = await invoke("get_available_models");
      // Only show downloadable models for onboarding
      setAvailableModels(models.filter((m) => !m.is_downloaded));
    } catch (err) {
      console.error("Failed to load models:", err);
      setError("Failed to load available models");
    }
  };

  const handleDownloadModel = async (modelId: string) => {
    setDownloading(true);
    setError(null);

    // Immediately transition to main app - download will continue in footer
    onModelSelected();

    try {
      await invoke("download_model", { modelId });
    } catch (err) {
      console.error("Download failed:", err);
      setError(`Failed to download model: ${err}`);
      setDownloading(false);
    }
  };

  const getRecommendedBadge = (modelId: string): boolean => {
    return modelId === "qwen3-asr";
  };

  return (
    <div className="max-w-4xl mx-auto text-center space-y-8">
      <p className="text-text/70 max-w-md mx-auto">
        To get started, choose a transcription model to download
      </p>

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-4 sm:gap-6 max-w-2xl mx-auto">
        {availableModels
          .sort((a, b) => a.size_mb - b.size_mb)
          .map((model) => (
            <button
              key={model.id}
              onClick={() => handleDownloadModel(model.id)}
              disabled={downloading}
              className="relative border-2 border-mid-gray/20 rounded-xl p-4 sm:p-6 text-left hover:border-logo-primary/50 hover:bg-logo-primary/5 hover:shadow-lg hover:scale-[1.02] focus:border-logo-primary focus:outline-none focus:ring-2 focus:ring-logo-primary/25 active:scale-[0.98] transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:border-mid-gray/20 disabled:hover:bg-transparent disabled:hover:shadow-none disabled:hover:scale-100 cursor-pointer group"
            >
              {getRecommendedBadge(model.id) && (
                <div className="absolute -top-2 -right-2 bg-logo-primary text-white text-xs px-3 py-1 rounded-full font-medium shadow-md">
                  Recommended
                </div>
              )}

              <div className="space-y-3">
                <div className="space-y-2">
                  <h3 className="text-lg sm:text-xl font-semibold text-text group-hover:text-logo-primary transition-colors">
                    {model.name}
                  </h3>
                  <p className="text-text/60 text-xs sm:text-sm leading-relaxed">
                    {model.description}
                  </p>
                </div>
              </div>
            </button>
          ))}
      </div>
    </div>
  );
};

export default Onboarding;
