"use client"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import {
  FileHeart,
  FileCode,
  Upload,
  Trees,
  Network,
  Fingerprint,
  AlertCircle,
  CheckCircle2,
  Loader2,
  ChevronRight,
  BarChart3,
  Activity,
  TrendingUp,
} from "lucide-react"
import { motion } from "framer-motion"

// Beat symbol meanings mapping
const beatMeanings = {
  "N": "Normal beat (displayed as \"·\" by the PhysioBank ATM, LightWAVE, pschart, and psfd)",
  "L": "Left bundle branch block beat",
  "R": "Right bundle branch block beat",
  "B": "Bundle branch block beat (unspecified)",
  "A": "Atrial premature beat",
  "a": "Aberrated atrial premature beat",
  "J": "Nodal (junctional) premature beat",
  "S": "Supraventricular premature or ectopic beat (atrial or nodal)",
  "V": "Premature ventricular contraction",
  "r": "R-on-T premature ventricular contraction",
  "F": "Fusion of ventricular and normal beat",
  "e": "Atrial escape beat",
  "j": "Nodal (junctional) escape beat",
  "n": "Supraventricular escape beat (atrial or nodal)",
  "E": "Ventricular escape beat",
  "/": "Paced beat",
  "f": "Fusion of paced and normal beat",
  "Q": "Unclassifiable beat",
  "?": "Beat not classified during learning"
}

// Anomalous beat types (non-normal beats)
const anomalousBeatTypes = ["L", "R", "A", "a", "J", "S", "V", "F", "-", "e", "E", "f", "x", "Q", "?", "U"]

export default function ECGAnalysis() {
  const [files, setFiles] = useState({
    header: null,
    signal: null,
  })
  const [selectedModel, setSelectedModel] = useState("")
  const [modelMetrics, setModelMetrics] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [results, setResults] = useState(null)
  const [activeTab, setActiveTab] = useState("upload")
  const audioRef = useRef(null)

  // Fetch model metrics from API
  const fetchModelMetrics = async () => {
    try {
      const response = await fetch("http://localhost:8000/metrics")
      const data = await response.json()
      setModelMetrics(data.results)
    } catch (error) {
      console.error("Failed to fetch model metrics:", error)
    }
  }

  // Load metrics when component mounts
  useState(() => {
    fetchModelMetrics()
  }, [])

  const handleFileChange = (type, file) => {
    setFiles((prev) => ({
      ...prev,
      [type]: file,
    }))
  }

  const handleDrop = (type, e) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(type, e.dataTransfer.files[0])
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
  }

  const validateFiles = () => {
    const { header, signal } = files

    if (!header || !signal) {
      return false
    }

    const headerExt = header.name.split(".").pop().toLowerCase()
    const signalExt = signal.name.split(".").pop().toLowerCase()

    return headerExt === "hea" && signalExt === "dat"
  }

  // Process beat analysis results
  const processBeatResults = (outputOfModel) => {
    // Count beat frequencies
    const beatCounts = {}
    outputOfModel.forEach((beat) => {
      // Treat both "N" and "." as normal beats
      const normalizedBeat = beat === "." ? "N" : beat
      beatCounts[normalizedBeat] = (beatCounts[normalizedBeat] || 0) + 1
    })

    // Sort by frequency (descending)
    const sortedBeats = Object.entries(beatCounts).sort(([, a], [, b]) => b - a)

    // Detect anomalies
    const totalBeats = outputOfModel.length
    const normalBeats = (beatCounts["N"] || 0) + (beatCounts["."] || 0)
    const anomalousBeats = sortedBeats.filter(
      ([symbol]) => anomalousBeatTypes.includes(symbol) && beatCounts[symbol] > 0,
    )

    const anomalyDetected = anomalousBeats.length > 0
    const anomalyPercentage = (((totalBeats - normalBeats) / totalBeats) * 100).toFixed(2)

    return {
      beatCounts: sortedBeats,
      totalBeats,
      normalBeats,
      anomalousBeats,
      anomalyDetected,
      anomalyPercentage,
    }
  }













  const analyzeECG = async () => {
    if (!validateFiles() || !selectedModel) return

    setIsAnalyzing(true)
    setProgress(0)

    try {
      const formData = new FormData()
      formData.append("dat", files.signal)
      formData.append("hea", files.header)
      formData.append("model_name", selectedModel)

      // Simulate progress
      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(interval)
            return prev
          }
          return prev + 10
        })
      }, 500)

      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      })

      const data = await response.json()

      clearInterval(interval)
      setProgress(100)

      // Process the results
      const processedResults = processBeatResults(data.result.OutputOfModel)

      setResults({
        ...data,
        analysis: processedResults,
      })

      setIsAnalyzing(false)
      setActiveTab("results")
    } catch (error) {
      console.error("Analysis failed:", error)
      setIsAnalyzing(false)
      // You might want to show an error message to the user here
    }
  }

  const allFilesUploaded = files.header && files.signal
  const isReadyToAnalyze = allFilesUploaded && selectedModel && validateFiles()

  const getModelDisplayName = (modelKey) => {
    const modelNames = {
      "Decision Tree": "Decision Tree",
      "Random Forest": "Random Forest",
      SVM: "SVM",
      KNN: "KNN",
    }
    return modelNames[modelKey] || modelKey
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      <div className="container mx-auto py-8 px-4">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-slate-800 dark:text-slate-100 mb-2">ECG Anomaly Detection</h1>
          <p className="text-slate-600 dark:text-slate-400 max-w-2xl mx-auto">
            Upload your ECG files, select a model, and get detailed analysis of cardiac anomalies
          </p>
        </motion.div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="max-w-6xl mx-auto">
          <TabsList className="grid grid-cols-3 mb-8">
            <TabsTrigger value="upload" className="data-[state=active]:bg-blue-500 data-[state=active]:text-white">
              1. Upload Files
            </TabsTrigger>
            <TabsTrigger
              value="model"
              disabled={!allFilesUploaded}
              className="data-[state=active]:bg-blue-500 data-[state=active]:text-white"
            >
              2. Select Model
            </TabsTrigger>
            <TabsTrigger
              value="results"
              disabled={!results}
              className="data-[state=active]:bg-blue-500 data-[state=active]:text-white"
            >
              3. View Results
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="mt-0">
            <div className="grid md:grid-cols-2 gap-6">
              {/* Header File Upload */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.1 }}
              >
                <Card className="h-full">
                  <CardHeader className="pb-4">
                    <CardTitle className="flex items-center gap-2">
                      <FileHeart className="h-5 w-5 text-red-500" />
                      Header File
                    </CardTitle>
                    <CardDescription>
                      Upload a *.hea file containing metadata (channels, sampling frequency, etc.)
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div
                      className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
                        files.header
                          ? "border-green-500 bg-green-50 dark:bg-green-900/20"
                          : "border-slate-300 hover:border-blue-400 dark:border-slate-700"
                      }`}
                      onDrop={(e) => handleDrop("header", e)}
                      onDragOver={handleDragOver}
                    >
                      {files.header ? (
                        <div className="flex flex-col items-center gap-2">
                          <CheckCircle2 className="h-8 w-8 text-green-500" />
                          <p className="text-sm font-medium">{files.header.name}</p>
                          <Button variant="outline" size="sm" onClick={() => handleFileChange("header", null)}>
                            Change File
                          </Button>
                        </div>
                      ) : (
                        <>
                          <Upload className="h-8 w-8 mx-auto mb-2 text-slate-400" />
                          <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                            Drag & drop your header file here
                          </p>
                          <label htmlFor="header-upload">
                            <Button variant="outline" size="sm" className="cursor-pointer" asChild>
                              <span>Browse Files</span>
                            </Button>
                            <input
                              id="header-upload"
                              type="file"
                              accept=".hea"
                              className="hidden"
                              onChange={(e) => {
                                if (e.target.files && e.target.files[0]) {
                                  handleFileChange("header", e.target.files[0])
                                }
                              }}
                            />
                          </label>
                        </>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {/* Signal File Upload */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, delay: 0.2 }}
              >
                <Card className="h-full">
                  <CardHeader className="pb-4">
                    <CardTitle className="flex items-center gap-2">
                      <FileCode className="h-5 w-5 text-blue-500" />
                      Signal File
                    </CardTitle>
                    <CardDescription>Upload a *.dat binary file containing raw ECG data</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div
                      className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors ${
                        files.signal
                          ? "border-green-500 bg-green-50 dark:bg-green-900/20"
                          : "border-slate-300 hover:border-blue-400 dark:border-slate-700"
                      }`}
                      onDrop={(e) => handleDrop("signal", e)}
                      onDragOver={handleDragOver}
                    >
                      {files.signal ? (
                        <div className="flex flex-col items-center gap-2">
                          <CheckCircle2 className="h-8 w-8 text-green-500" />
                          <p className="text-sm font-medium">{files.signal.name}</p>
                          <Button variant="outline" size="sm" onClick={() => handleFileChange("signal", null)}>
                            Change File
                          </Button>
                        </div>
                      ) : (
                        <>
                          <Upload className="h-8 w-8 mx-auto mb-2 text-slate-400" />
                          <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                            Drag & drop your signal file here
                          </p>
                          <label htmlFor="signal-upload">
                            <Button variant="outline" size="sm" className="cursor-pointer" asChild>
                              <span>Browse Files</span>
                            </Button>
                            <input
                              id="signal-upload"
                              type="file"
                              accept=".dat"
                              className="hidden"
                              onChange={(e) => {
                                if (e.target.files && e.target.files[0]) {
                                  handleFileChange("signal", e.target.files[0])
                                }
                              }}
                            />
                          </label>
                        </>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            </div>

            <div className="mt-6 flex justify-end">
              <Button onClick={() => setActiveTab("model")} disabled={!allFilesUploaded} className="gap-2">
                Next Step <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </TabsContent>

          <TabsContent value="model" className="mt-0">
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
              <h2 className="text-2xl font-bold mb-6 text-center">Select Analysis Model</h2>

              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {/* Decision Tree Model */}
                <Card
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedModel === "Decision Tree" ? "ring-2 ring-blue-500 dark:ring-blue-400" : ""
                  }`}
                  onClick={() => setSelectedModel("Decision Tree")}
                >
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Fingerprint className="h-5 w-5 text-blue-500" />
                      Decision Tree
                    </CardTitle>
                    <CardDescription>Tree-based Classification</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
                      Simple, interpretable model with clear decision paths.
                    </p>
                    {modelMetrics && modelMetrics["Decision Tree"] && (
                      <div className="space-y-2 text-xs">
                        <div className="flex justify-between">
                          <span>Accuracy:</span>
                          <span className="font-semibold">
                            {(modelMetrics["Decision Tree"].Accuracy * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Precision:</span>
                          <span className="font-semibold">
                            {(modelMetrics["Decision Tree"].Precision * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Recall:</span>
                          <span className="font-semibold">
                            {(modelMetrics["Decision Tree"].Recall * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>F1-Score:</span>
                          <span className="font-semibold">
                            {(modelMetrics["Decision Tree"]["F1-Score"] * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    )}
                  </CardContent>
                  <CardFooter>
                    <Badge variant={selectedModel === "Decision Tree" ? "default" : "outline"}>
                      {selectedModel === "Decision Tree" ? "Selected" : "Select"}
                    </Badge>
                  </CardFooter>
                </Card>

                {/* Random Forest Model */}
                <Card
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedModel === "Random Forest" ? "ring-2 ring-blue-500 dark:ring-blue-400" : ""
                  }`}
                  onClick={() => setSelectedModel("Random Forest")}
                >
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Trees className="h-5 w-5 text-green-500" />
                      Random Forest
                    </CardTitle>
                    <CardDescription>Ensemble Learning Method</CardDescription>
                  </CardHeader>


                  <CardContent>
                    <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
                      Better robustness and accuracy. Handles complex data well.
                    </p>
                    {modelMetrics && modelMetrics["Random Forest"] && (
                      <div className="space-y-2 text-xs">
                        <div className="flex justify-between">
                          <span>Accuracy:</span>
                          <span className="font-semibold">
                            {(modelMetrics["Random Forest"].Accuracy * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Precision:</span>
                          <span className="font-semibold">
                            {(modelMetrics["Random Forest"].Precision * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Recall:</span>
                          <span className="font-semibold">
                            {(modelMetrics["Random Forest"].Recall * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>F1-Score:</span>
                          <span className="font-semibold">
                            {(modelMetrics["Random Forest"]["F1-Score"] * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    )}
                  </CardContent>
                  <CardFooter>
                    <Badge variant={selectedModel === "Random Forest" ? "default" : "outline"}>
                      {selectedModel === "Random Forest" ? "Selected" : "Select"}
                    </Badge>
                  </CardFooter>
                </Card>

                 {/* KNN Model */}
                <Card
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedModel === "KNN" ? "ring-2 ring-blue-500 dark:ring-blue-400" : ""
                  }`}
                  onClick={() => setSelectedModel("KNN")}
                >
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5 text-indigo-500" />
                      KNN
                    </CardTitle>
                    <CardDescription>distance-based learning Method </CardDescription>
                  </CardHeader>

                  
                  <CardContent>
                    <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
                       Works well with small datasets with clear separation between classes .
                    </p>
                    {modelMetrics && modelMetrics["KNN"] && (
                      <div className="space-y-2 text-xs">
                        <div className="flex justify-between">
                          <span>Accuracy:</span>
                          <span className="font-semibold">
                            {(modelMetrics["KNN"].Accuracy * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Precision:</span>
                          <span className="font-semibold">
                            {(modelMetrics["KNN"].Precision * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>Recall:</span>
                          <span className="font-semibold">
                            {(modelMetrics["KNN"].Recall * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span>F1-Score:</span>
                          <span className="font-semibold">
                            {(modelMetrics["KNN"]["F1-Score"] * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    )}
                  </CardContent>
                  <CardFooter>
                    <Badge variant={selectedModel === "KNN" ? "default" : "outline"}>
                      {selectedModel === "KNN" ? "Selected" : "Select"}
                    </Badge>
                  </CardFooter>
                </Card>

                {/* SVM Model */}
                <Card
                  className={`cursor-pointer transition-all hover:shadow-md ${
                    selectedModel === "SVM" ? "ring-2 ring-blue-500 dark:ring-blue-400" : ""
                  }`}
                  onClick={() => setSelectedModel("SVM")}
                >
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Network className="h-5 w-5 text-orange-500" />
                      SVM
                    </CardTitle>
                    <CardDescription>Support Vector Machine</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-slate-600 dark:text-slate-400 mb-4">
                      Excellent for binary classification tasks with clear margins.
                    </p>
                    {modelMetrics && modelMetrics["SVM"] && (
                      <div className="space-y-2 text-xs">
                        <div className="flex justify-between">
                          <span>Accuracy:</span>
                          <span className="font-semibold">{(modelMetrics["SVM"].Accuracy * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Precision:</span>
                          <span className="font-semibold">{(modelMetrics["SVM"].Precision * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Recall:</span>
                          <span className="font-semibold">{(modelMetrics["SVM"].Recall * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span>F1-Score:</span>
                          <span className="font-semibold">{(modelMetrics["SVM"]["F1-Score"] * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    )}
                  </CardContent>
                  <CardFooter>
                    <Badge variant={selectedModel === "SVM" ? "default" : "outline"}>
                      {selectedModel === "SVM" ? "Selected" : "Select"}
                    </Badge>
                  </CardFooter>
                </Card>
              </div>

              <div className="mt-8 flex justify-between">
                <Button variant="outline" onClick={() => setActiveTab("upload")}>
                  Back
                </Button>

                <Button onClick={analyzeECG} disabled={!isReadyToAnalyze || isAnalyzing} className="gap-2">
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    "Analyze ECG"
                  )}
                </Button>
              </div>

              {isAnalyzing && (
                <div className="mt-8">
                  <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                    Analyzing ECG data using {selectedModel} model...
                  </p>
                  <Progress value={progress} className="h-2" />
                  <p className="text-xs text-slate-500 dark:text-slate-500 mt-2 text-right">{progress}% complete</p>
                </div>
              )}
            </motion.div>
          </TabsContent>

          <TabsContent value="results" className="mt-0">
            {results && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.5 }}>
                <div className="space-y-8">
                  {/* Analysis Summary */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Activity className="h-5 w-5 text-blue-500" />
                        Analysis Summary
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid md:grid-cols-4 gap-4">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-blue-600">{results.analysis.totalBeats}</div>
                          <div className="text-sm text-slate-600">Total Beats</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-green-600">{results.analysis.normalBeats}</div>
                          <div className="text-sm text-slate-600">Normal Beats</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-orange-600">
                            {results.analysis.anomalousBeats.length}
                          </div>
                          <div className="text-sm text-slate-600">Anomaly Types</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-red-600">{results.analysis.anomalyPercentage}%</div>
                          <div className="text-sm text-slate-600">Anomaly Rate</div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Anomaly Detection Result */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <TrendingUp className="h-5 w-5 text-purple-500" />
                        Anomaly Detection Result
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      {results.analysis.anomalyDetected ? (
                        <Alert className="bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800">
                          <AlertCircle className="h-4 w-4 text-red-600 dark:text-red-400" />
                          <AlertTitle className="text-red-800 dark:text-red-300">Anomalies Detected</AlertTitle>
                          <AlertDescription className="text-red-700 dark:text-red-300">
                            The following anomalous beat types were detected:{" "}
                            <span className="font-semibold">
                              {results.analysis.anomalousBeats.map(([symbol]) => symbol).join(", ")}
                            </span>
                          </AlertDescription>
                        </Alert>
                      ) : (
                        <Alert className="bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800">
                          <CheckCircle2 className="h-4 w-4 text-green-600 dark:text-green-400" />
                          <AlertTitle className="text-green-800 dark:text-green-300">No Anomalies Detected</AlertTitle>
                          <AlertDescription className="text-green-700 dark:text-green-300">
                            The ECG analysis shows predominantly normal beats with no significant anomalies.
                          </AlertDescription>
                        </Alert>
                      )}
                    </CardContent>
                  </Card>

                  {/* Beat Frequency Analysis */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <BarChart3 className="h-5 w-5 text-green-500" />
                        Beat Frequency Analysis
                      </CardTitle>
                      <CardDescription>Beat counts in descending order of frequency</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="grid md:grid-cols-2 gap-6">
                        <div>
                          <h4 className="font-semibold mb-3">Beat Frequencies</h4>
                          <div className="space-y-2">
                            {results.analysis.beatCounts.slice(0, 10).map(([symbol, count]) => (
                              <div key={symbol} className="flex justify-between items-center">
                                <span className="font-mono text-sm">{symbol}</span>
                                <div className="flex items-center gap-2">
                                  <div className="w-20 bg-slate-200 dark:bg-slate-700 rounded-full h-2">
                                    <div
                                      className="bg-blue-500 h-2 rounded-full"
                                      style={{ width: `${(count / results.analysis.totalBeats) * 100}%` }}
                                    />
                                  </div>
                                  <span className="text-sm font-semibold w-12 text-right">{count}</span>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                       
                      </div>
                    </CardContent>
                  </Card>

                  {/* Beat Symbol Reference Table */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Beat Symbol Reference</CardTitle>
                      <CardDescription>Complete mapping of beat symbols to their medical meanings</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead className="w-20">Symbol</TableHead>
                            <TableHead>Meaning</TableHead>
                            <TableHead className="w-24">Status</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {Object.entries(beatMeanings).map(([symbol, meaning]) => (
                            <TableRow key={symbol}>
                              <TableCell className="font-mono font-semibold">{symbol}</TableCell>
                              <TableCell>{meaning}</TableCell>
                              <TableCell>
                                <Badge variant={symbol === "N" || symbol === "." ? "default" : "destructive"}>
                                  {symbol === "N" || symbol === "." ? "Normal" : "Anomaly"}
                                </Badge>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>

                  <div className="flex justify-between">
                    <Button variant="outline" onClick={() => setActiveTab("model")}>
                      Back to Model Selection
                    </Button>

                    <Button
                      onClick={() => {
                        setFiles({ header: null, signal: null })
                        setSelectedModel("")
                        setResults(null)
                        setActiveTab("upload")
                      }}
                    >
                      Start New Analysis
                    </Button>
                  </div>
                </div>
              </motion.div>
            )}
          </TabsContent>
        </Tabs>
      </div>

      {/* Audio element for playing anomaly sounds */}
      <audio ref={audioRef} className="hidden" />
    </div>
  )
}
