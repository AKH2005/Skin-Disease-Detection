const API_URL = "https://akhil23bce-skindetectionbackend.hf.space"

let predictionData = null
let latestPdfUrl = null

function goTo(id) {
  document.querySelectorAll("section").forEach(sec => sec.classList.add("hidden"))
  const target = document.getElementById(id)
  if (target) target.classList.remove("hidden")
}

function previewImage(event) {
  const file = event.target.files[0]
  if (!file) return

  const preview = document.getElementById("preview")

  if (preview.src && preview.src.startsWith("blob:")) {
    URL.revokeObjectURL(preview.src)
  }

  if (!file.type.startsWith("image/")) {
    showError("upload-error", "Please select a valid image.")
    event.target.value = ""
    preview.classList.add("hidden")
    return
  }

  if (file.size > 5 * 1024 * 1024) {
    showError("upload-error", "Please upload an image smaller than 5MB.")
    event.target.value = ""
    preview.classList.add("hidden")
    return
  }

  hideError("upload-error")
  preview.src = URL.createObjectURL(file)
  preview.classList.remove("hidden")
}

async function analyzeImage() {
  const file = document.getElementById("image-input").files[0]

  if (!file) {
    showError("upload-error", "Please upload an image before analyzing.")
    return
  }

  predictionData = null
  latestPdfUrl = null

  setAnalyzeButtonState(true)
  hideError("upload-error")
  hideError("report-error")
  goTo("loading")

  const formData = new FormData()
  formData.append("image", file)

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 120000)

  try {
    const res = await fetch(`${API_URL}/api/predict`, {
      method: "POST",
      body: formData,
      signal: controller.signal
    })

    clearTimeout(timeout)

    if (!res.ok) {
      let errMsg = `Server error ${res.status}`
      try {
        const errData = await res.json()
        if (errData.error) errMsg = errData.error
      } catch (_) {}
      throw new Error(errMsg)
    }

    const data = await res.json()
    if (data.error) {
      showError("upload-error", data.error)
      goTo("upload")
      return
    }

    predictionData = data
    renderResult(data)

    // FIX: show both results and care guide without goTo hiding one of them
    document.querySelectorAll("section").forEach(sec => sec.classList.add("hidden"))
    document.getElementById("results").classList.remove("hidden")
    document.getElementById("ai-care-guide").classList.remove("hidden")
    window.scrollTo({ top: 0, behavior: "smooth" })

  } catch (err) {
    clearTimeout(timeout)
    console.error("Prediction error:", err)
    showError("upload-error", "Prediction failed: " + err.message)
    goTo("upload")
  } finally {
    setAnalyzeButtonState(false)
  }
}

function setList(id, items) {
  const el = document.getElementById(id)
  if (!el) return
  el.innerHTML = ""
  ;(items || []).forEach(item => {
    const li = document.createElement("li")
    li.textContent = item
    el.appendChild(li)
  })
}

function setText(id, value) {
  const el = document.getElementById(id)
  if (!el) return
  el.textContent = value || "Not available"
}

function renderResult(data) {
  document.getElementById("result-disease").innerText = "Disease: " + data.disease
  document.getElementById("ai-summary").innerText = data.ai_summary || "No AI summary available."

  const confidence = Number(data.confidence || 0)
  document.getElementById("confidence-bar").style.width = `${Math.min(confidence, 100)}%`
  document.getElementById("confidence-text").innerText = `Confidence: ${confidence.toFixed(2)}%`

  const warningBox = document.getElementById("warning-box")
  if (data.warning) {
    warningBox.textContent = data.warning
    warningBox.classList.remove("hidden")
  } else {
    warningBox.textContent = ""
    warningBox.classList.add("hidden")
  }

  const guide = data.care_guide || {}

  setText("care-overview", guide.overview)
  setText("care-simple", guide.simple_explanation)
  setList("care-symptoms", guide.common_symptoms)
  setList("care-early", guide.early_symptoms)
  setList("care-advanced", guide.advanced_symptoms)
  setList("care-causes", guide.possible_causes)
  setList("care-risk-factors", guide.risk_factors)
  setText("care-type", guide.condition_type)
  setText("care-contagious", guide.contagious)
  setText("care-severity", guide.severity_level)
  setList("care-home", guide.safe_home_care)
  setList("care-routine", guide.daily_skin_routine)
  setList("care-avoid", guide.what_to_avoid)
  setList("care-hygiene", guide.hygiene_advice)
  setList("care-sun", guide.sun_exposure_advice)
  setList("care-clothing", guide.clothing_advice)
  setList("care-stress", guide.stress_sleep_advice)
  setList("care-foods-good", guide.foods_to_eat)
  setList("care-foods-bad", guide.foods_to_avoid)
  setList("care-hydration", guide.hydration_advice)
  setList("care-vitamins", guide.general_vitamins)
  setList("care-doctor", guide.when_to_see_doctor)
  setList("care-warning", guide.urgent_warning_signs)
  setList("care-emergency", guide.emergency_red_flags)
  setList("care-work", guide.work_school_precautions)
  setList("care-exercise", guide.exercise_precautions)
  setList("care-seasonal", guide.seasonal_triggers)
  setText("care-specialist", guide.doctor_specialist)
  setText("care-summary", guide.report_summary)
}

async function generatePdfIfNeeded(openAfterGenerate = false) {
  if (!predictionData) {
    showError("report-error", "Please analyze an image first.")
    return null
  }

  const name = document.getElementById("name").value.trim()
  const age = document.getElementById("age").value.trim()
  const gender = document.getElementById("gender").value.trim()

  if (!name || !age || !gender) {
    showError("report-error", "Please fill in Patient Name, Age, and Gender.")
    return null
  }

  hideError("report-error")

  try {
    const res = await fetch(`${API_URL}/api/save-prescription`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name,
        age,
        gender,
        // FIX: removed undefined `email` and `whatsapp` variables
        raw_disease: predictionData.raw_disease,
        confidence: predictionData.confidence
      })
    })

    const data = await res.json()

    if (!res.ok || data.error) {
      showError("report-error", data.error || "Failed to generate PDF report.")
      return null
    }

    latestPdfUrl = data.pdf_url

    if (openAfterGenerate) {
      window.open(latestPdfUrl, "_blank")
    }

    return latestPdfUrl
  } catch (err) {
    console.error("PDF generation error:", err)
    showError("report-error", "Failed to generate PDF report.")
    return null
  }
}

async function downloadPDFReport() {
  const btn = document.getElementById("download-pdf-btn")
  const originalText = btn.textContent
  btn.disabled = true
  btn.textContent = "Preparing..."

  try {
    await generatePdfIfNeeded(true)
  } finally {
    btn.disabled = false
    btn.textContent = originalText
  }
}

async function shareReport() {
  const btn = document.getElementById("share-report-btn")
  const originalText = btn.textContent
  btn.disabled = true
  btn.textContent = "Preparing..."

  try {
    const pdfUrl = await generatePdfIfNeeded(false)
    if (!pdfUrl) return

    const name = document.getElementById("name").value.trim()
    const shareTitle = "DermAI Skin Report"
    const shareText =
`Hello ${name},

Your DermAI report is ready.

Disease: ${predictionData.disease}
Confidence: ${Number(predictionData.confidence || 0).toFixed(2)}%

Please consult a dermatologist for professional diagnosis.`

    if (navigator.share) {
      try {
        await navigator.share({
          title: shareTitle,
          text: shareText,
          url: pdfUrl
        })
        return
      } catch (shareErr) {
        if (shareErr.name === "AbortError") return
      }
    }

    try {
      await navigator.clipboard.writeText(pdfUrl)
      alert("Report link copied to clipboard.")
    } catch (_) {
      window.open(pdfUrl, "_blank")
      alert("PDF opened. Share manually.")
    }

  } finally {
    btn.disabled = false
    btn.textContent = originalText
  }
}

function downloadTextReport() {
  if (!predictionData) {
    alert("No prediction available.")
    return
  }

  const guide = predictionData.care_guide || {}

  let text = `DermAI 2.0 Report\n`
  text += `${"=".repeat(40)}\n\n`
  text += `Disease: ${predictionData.disease}\n`
  text += `Confidence: ${Number(predictionData.confidence || 0).toFixed(2)}%\n\n`
  text += `AI Summary:\n${predictionData.ai_summary || ""}\n\n`
  text += `Overview:\n${guide.overview || ""}\n\n`
  text += `Simple Explanation:\n${guide.simple_explanation || ""}\n\n`

  const sections = [
    ["Common Symptoms", guide.common_symptoms],
    ["Early Symptoms", guide.early_symptoms],
    ["Advanced Symptoms", guide.advanced_symptoms],
    ["Possible Causes", guide.possible_causes],
    ["Risk Factors", guide.risk_factors],
    ["Safe Home Care", guide.safe_home_care],
    ["What to Avoid", guide.what_to_avoid],
    ["Foods to Eat", guide.foods_to_eat],
    ["Foods to Avoid", guide.foods_to_avoid],
    ["When to See a Doctor", guide.when_to_see_doctor],
    ["Urgent Warning Signs", guide.urgent_warning_signs]
  ]

  sections.forEach(([title, items]) => {
    text += `${title}:\n`
    ;(items || []).forEach(i => text += `- ${i}\n`)
    text += `\n`
  })

  text += `⚠️ This is AI-generated guidance, not a final medical diagnosis.\n`

  const blob = new Blob([text], { type: "text/plain" })
  const link = document.createElement("a")
  link.href = URL.createObjectURL(blob)
  link.download = "DermAI_Advanced_Report.txt"
  link.click()
  URL.revokeObjectURL(link.href)
}

function findDoctors() {
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(
      pos => {
        const { latitude, longitude } = pos.coords
        window.open(`https://www.google.com/maps/search/dermatologist/@${latitude},${longitude},13z`, "_blank")
      },
      () => {
        window.open("https://www.google.com/maps/search/dermatologist+near+me", "_blank")
      }
    )
  } else {
    window.open("https://www.google.com/maps/search/dermatologist+near+me", "_blank")
  }
}

function showError(id, message) {
  const el = document.getElementById(id)
  if (el) {
    el.textContent = message
    el.classList.remove("hidden")
  }
}

function hideError(id) {
  const el = document.getElementById(id)
  if (el) {
    el.textContent = ""
    el.classList.add("hidden")
  }
}

function setAnalyzeButtonState(loading) {
  const btn = document.getElementById("analyze-btn")
  if (!btn) return
  btn.disabled = loading
  btn.textContent = loading ? "Analyzing..." : "Analyze Image"
}