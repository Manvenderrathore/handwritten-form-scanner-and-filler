from flask import Flask, request, render_template_string, jsonify, send_file
import cv2, os, re, json, csv, io
from datetime import datetime
from pathlib import Path
from rapidocr_onnxruntime import RapidOCR
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

STATS_FILE = 'stats.json'
CSV_PATH = 'submissions.csv'
HISTORY_LIMIT = 200


DEFAULT_STATS = {
    "total": 0,
    "fields": {"name": 0, "email": 0, "phone": 0, "gpa": 0, "major": 0, "date_of_birth": 0},
    "history": []
}

if Path(STATS_FILE).is_file():
    try:
        with open(STATS_FILE, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        stats.setdefault("total", 0)
        stats.setdefault("fields", DEFAULT_STATS["fields"].copy())
        stats.setdefault("history", [])
    except Exception:
        stats = DEFAULT_STATS.copy()
else:
    stats = DEFAULT_STATS.copy()

reader = RapidOCR()


def enhance(path):
    """Enhance image for better OCR."""
    img = cv2.imread(path)
    if img is None:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(path, thresh)


def extract_data(path):
    enhance(path)
    results, _ = reader(path)
    full_text = " ".join([t[1] for t in results]).strip() if results else ""

    extracted = {
        "name": "Not found",
        "email": "Not found",
        "phone": "Not found",
        "gpa": "Not found",
        "major": "Not found",
        "date_of_birth": "Not found"
    }
    found = {k: 0 for k in extracted.keys()}

    name_candidate = None
    m_name_label = re.search(r'\bname[:\s\-]{1,6}([A-Za-zÀ-ÖØ-öø-ÿ \'\.-]{2,120})', full_text, re.I)
    if m_name_label:
        candidate = m_name_label.group(1).strip()
        candidate = re.split(r'\b(email|phone|gpa|major|dob|date of birth)\b', candidate, flags=re.I)[0].strip()
        if candidate:
            name_candidate = candidate

    if not name_candidate:
        m_title = re.search(r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,4})\b', full_text)
        if m_title:
            name_candidate = m_title.group(1)

    if not name_candidate:
        m_allcaps = re.search(r'\b([A-Z]{2,}(?:\s[A-Z]{2,}){1,4})\b', full_text)
        if m_allcaps:
            name_candidate = m_allcaps.group(1).title()

    if name_candidate and not any(w in name_candidate.lower() for w in ['major', 'email', 'phone', 'gpa', 'date', 'dob', 'birth']):
        extracted['name'] = ' '.join(name_candidate.split())
        found['name'] = 1

    clean_email = full_text.lower().replace(' ', '')
    clean_email = clean_email.replace('0', 'o').replace('1', 'l')
    email_re = re.search(r'([a-z0-9._%+\-]+@[a-z0-9.\-]+\.(?:com|edu|org|net|gov|in|co|info|io|me|tech|ai|[a-z]{2,6}))', clean_email, re.I)
    if email_re:
        extracted['email'] = email_re.group(1)
        found['email'] = 1

    phone_re = re.search(r'(\+?\d{1,3}[\s\-\.])?(?:\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4})', full_text)
    if phone_re:
        extracted['phone'] = phone_re.group(0).strip()
        found['phone'] = 1

    gpa_re = re.search(r'\b([0-4]\.[0-9]{1,2})\b', full_text)
    if gpa_re:
        extracted['gpa'] = gpa_re.group(1)
        found['gpa'] = 1

    dob_re = re.search(r'\b(\d{1,2}[\/\-\.\s]\d{1,2}[\/\-\.\s]\d{2,4})\b', full_text)
    if dob_re:
        extracted['date_of_birth'] = dob_re.group(1)
        found['date_of_birth'] = 1

    major_re = re.search(r'\bmajor[:\s\-]{0,6}([A-Za-z &/]{2,80})', full_text, re.I)
    if major_re:
        val = major_re.group(1).strip()
        val = re.split(r'\b(email|phone|gpa|dob|date of birth|name)\b', val, flags=re.I)[0].strip()
        if val:
            extracted['major'] = val.title()
            found['major'] = 1

    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Name', 'Email', 'Phone', 'GPA', 'Major', 'Date of Birth'])

    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            extracted["name"],
            extracted["email"],
            extracted["phone"],
            extracted["gpa"],
            extracted["major"],
            extracted["date_of_birth"]
        ])

    entry = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "found": found,
        "values": extracted.copy()
    }

    stats["total"] += 1
    for k, v in found.items():
        if v:
            stats["fields"][k] += 1

    stats["history"].append(entry)
    if len(stats["history"]) > HISTORY_LIMIT:
        stats["history"] = stats["history"][-HISTORY_LIMIT:]

    try:
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return extracted, found

HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Form OCR - FULL DASHBOARD</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body{font-family:system-ui;background:#f8f9fa;padding:18px;}
  .box{max-width:1200px;margin:auto;background:white;padding:24px;border-radius:12px;box-shadow:0 8px 24px #0001;}
  h1{color:#6b46c1;text-align:center;margin:6px 0 18px;}
  .grid{display:grid;grid-template-columns:1fr 420px;gap:20px;}
  .upload{border:3px dashed #8b5cf6;padding:30px;text-align:center;border-radius:12px;background:#f3e8ff;cursor:pointer;}
  .btn{background:#8b5cf6;color:white;padding:12px 28px;border:none;border-radius:999px;font-size:15px;cursor:pointer;margin-top:8px;}
  .preview{max-width:100%;border-radius:8px;margin-top:12px;display:none;}
  .field{background:#f3e8ff;padding:12px;border-radius:8px;margin:6px 0;border-left:6px solid #8b5cf6;}
  .small{font-size:13px;color:#666;}
  .row{display:flex;gap:12px;align-items:center;}
  table{width:100%;border-collapse:collapse;margin-top:12px;}
  th,td{padding:8px;border-bottom:1px solid #eee;text-align:left;font-size:13px;}
  .chart-card{background:#fff;padding:12px;border-radius:8px;box-shadow:0 4px 12px #0001;margin-bottom:12px;}
</style>
</head>
<body>
<div class="box">
  <h1>Form OCR — Full Analytics Dashboard</h1>
  <div class="grid">
    <div>
      <h3>Upload / Extract</h3>
      <div class="upload" onclick="document.getElementById('f').click()">
        Click to upload image<br><input type="file" id="f" accept="image/*" hidden>
      </div>
      <img id="prev" class="preview">
      <div style="text-align:center"><button class="btn" id="btn" onclick="go()" disabled>Extract Data</button></div>
      <div id="res" style="margin-top:14px"></div>

      <h3 style="margin-top:18px">Recent Uploads</h3>
     
<div style="margin: 20px 0; text-align: center;">
    <button onclick="downloadExcel()" 
            style="background:#10b981; color:white; padding:12px 30px; border:none; border-radius:50px; font-size:16px; cursor:pointer;">
        Download Excel Sheet (.xlsx)
    </button>
</div>
      <div id="recent-table"></div>
    </div>

    <div>
      <div class="chart-card">
        <h4>Last Image — Field Presence (0% / 100%)</h4>
        <canvas id="lastChart" height="120"></canvas>
      </div>

      <div class="chart-card">
        <h4>Aggregated Found % (Historical)</h4>
        <canvas id="historyChart" height="200"></canvas>
      </div>

      <div class="chart-card">
        <h4>Found vs Not Found (Aggregate)</h4>
        <canvas id="pieChart" height="140"></canvas>
      </div>
    </div>
  </div>
</div>

<script>
let lastChart, historyChart, pieChart;
const fields = ['name','email','phone','gpa','major','date_of_birth'];

document.getElementById('f').onchange = e=>{
  const file = e.target.files[0];
  if(file){
    document.getElementById('prev').src = URL.createObjectURL(file);
    document.getElementById('prev').style.display = 'block';
    document.getElementById('btn').disabled = false;
  }
};

async function go(){
  const file=document.getElementById('f').files[0];
  if(!file) return;
  const btn = document.getElementById('btn');
  btn.disabled = true; btn.textContent = "Processing...";
  const fd = new FormData(); fd.append('file', file);
  try{
    const r = await fetch('/', {method:'POST', body: fd});
    const d = await r.json();
    showResult(d);
    await refreshStats(); 
  }catch(err){
    console.error(err);
  } finally {
    btn.disabled = false; btn.textContent = "Extract Data";
  }
}

function showResult(d){

  let html = '<h3>Extracted</h3>';
  for(const f of fields){
    const val = d[f] || 'Not found';
    html += `<div class="field"><strong>${f.replace('_',' ')}:</strong> ${val}</div>`;
  }
  document.getElementById('res').innerHTML = html;
}

async function downloadExcel() {
    try {
        const response = await fetch('/download_excel');
        if (!response.ok) throw new Error("Failed to generate Excel");

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `form_submissions_${new Date().toISOString().slice(0,10)}.xlsx`;
        a.click();
        window.URL.revokeObjectURL(url);
    } catch (err) {
        alert("Error: " + err.message);
    }
}

async function refreshStats(){
  const r = await fetch('/stats?t=' + Date.now());
  const s = await r.json();
  const history = s.history || [];
  const last = history.length ? history[history.length - 1] : null;


  const lastData = fields.map(f => last ? (last.found[f] ? 100 : 0) : 0);
  if(lastChart) lastChart.destroy();
  lastChart = new Chart(document.getElementById('lastChart'), {
    type:'bar',
    data:{
      labels: fields.map(f=>f.replace('_',' ')),
      datasets:[{label:'Last Image', data:lastData}]
    },
    options:{indexAxis:'y',plugins:{legend:{display:false}},scales:{x:{min:0,max:100}}}
  });


  const windowSize = Math.min(history.length, 50); 
  const samples = history.slice(-windowSize);
 
  const labels = samples.map(e => new Date(e.ts).toLocaleTimeString());
  const datasets = fields.map((f, idx) => {
    const data = [];
    let cum = 0;
   
    for(let i=0;i<samples.length;i++){
      const slice = samples.slice(0, i+1);
      const foundCount = slice.reduce((acc, it) => acc + (it.found[f] ? 1 : 0), 0);
      const pct = Math.round((foundCount / (i+1)) * 100);
      data.push(pct);
    }
    return {label: f.replace('_',' '), data};
  });

  if(historyChart) historyChart.destroy();
  historyChart = new Chart(document.getElementById('historyChart'), {
    type:'line',
    data:{ labels, datasets },
    options:{ interaction:{mode:'index', intersect:false}, plugins:{legend:{position:'bottom'}}}
  });


  const aggFields = s.fields || {};
  const totalForms = s.total || 0;
  const foundCounts = fields.map(f => aggFields[f] || 0);
 
  const totalFound = foundCounts.reduce((a,b)=>a+b,0);
  const totalPossible = totalForms * fields.length;
  const totalNotFound = Math.max(0, totalPossible - totalFound);
  if(pieChart) pieChart.destroy();
  pieChart = new Chart(document.getElementById('pieChart'), {
    type:'pie',
    data:{ labels:['Found','Not Found'], datasets:[{data:[totalFound, totalNotFound]}] },
    options:{plugins:{legend:{position:'bottom'}}}
  });

 
  const recent = history.slice(-10).reverse();
  let table = '<table><thead><tr><th>Time (UTC)</th>';
  for(const f of fields) table += `<th>${f.replace('_',' ')}</th>`;
  table += '</tr></thead><tbody>';
  for(const row of recent){
    table += `<tr><td class="small">${new Date(row.ts).toLocaleString()}</td>`;
    for(const f of fields){
      table += `<td>${row.found[f] ? (row.values[f]||'Yes') : '<span class="small">Not found</span>'}</td>`;
    }
    table += '</tr>';
  }
  table += '</tbody></table>';
  document.getElementById('recent-table').innerHTML = table;
}


refreshStats();
setInterval(refreshStats, 3000); 
</script>
</body>
</html>""" 

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "no file"}), 400
        f = request.files['file']
        if not f.filename:
            return jsonify({"error": "no file selected"}), 400
        path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(path)
        try:
            extracted, found = extract_data(path)
        finally:
            if os.path.exists(path):
                os.remove(path)
        resp = extracted.copy()
        resp["_found"] = found
        return jsonify(resp)
    return render_template_string(HTML)


@app.route('/download_excel')
def download_excel():
    if not os.path.exists(CSV_PATH):
        return "No data yet", 400
    df = pd.read_csv(CSV_PATH)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Submissions')
    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name=f"Form_Submissions_{datetime.now().strftime('%Y-%m-%d')}.xlsx",
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


@app.route('/stats')
def get_stats():
    return jsonify(stats)


if __name__ == '__main__':
    print("OPEN → http://127.0.0.1:5000")
    app.run(port=5000, debug=False)
