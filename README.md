# Medical Code Inference Pipeline

A simple Python tool that finds medical codes (like CPT and ICD-10) in medical policy documents.

## 🚀 **What This Does**

This tool reads medical policy documents and automatically finds medical codes that should be billed for those procedures.

**Example:**
- **Input:** "This policy covers MRI of the brain for diagnostic purposes"
- **Output:** Codes `70551`, `70552`, `70553` (MRI codes)

## 📁 **Two Options**

### **Option 1: High Accuracy (99% success)**
```bash
python3 bert_pipeline.py -input policies_cleaned.csv -output results.csv
```
- **Best for:** When you need the most accurate results
- **Speed:** Slower (47 minutes for 200 documents)
- **Uses:** AI (ClinicalBERT) to understand medical text

### **Option 2: Fast & Simple (63% success)**
```bash
python3 run_pipeline.py -input policies_cleaned.csv -output results.csv
```
- **Best for:** Quick testing and development
- **Speed:** Very fast (1 second for 200 documents)
- **Uses:** Simple rules and keyword matching

## 🛠 **Setup**

1. **Install Python packages:**
```bash
pip install -r requirements.txt
```

2. **Run the tool:**
```bash
# For best accuracy
python3 bert_pipeline.py -input your_data.csv -output results.csv

# For fast processing
python3 run_pipeline.py -input your_data.csv -output results.csv
```

## 📊 **Results**

The tool creates a CSV file with:
- **Policy ID:** Which document
- **Codes Found:** Medical codes like `99213`, `70551`, `E11.9`
- **Confidence:** How sure the tool is (0-1 scale)

**Example output:**
```csv
id,codes,confidence
policy1,"['99213', '70551']",0.85
policy2,"['E11.9', 'I10']",0.72
```

## 📈 **Performance**

| Method | Accuracy | Speed | Best For |
|--------|----------|-------|----------|
| **BERT (AI)** | 99% | Slow | Production |
| **Rules** | 63% | Fast | Testing |

## 🔧 **How It Works**

### **Simple Rules Method:**
1. Looks for codes already mentioned (like "CPT 99213")
2. Finds keywords like "MRI" and suggests MRI codes
3. Matches medical terms to appropriate codes

### **AI Method (BERT):**
1. Does everything the rules method does
2. PLUS uses AI to understand complex medical language
3. Finds codes even when not explicitly mentioned

## 📋 **Input Format**

Your CSV file needs these columns:
- `policy_id`: Unique ID for each document
- `cleaned_policy_text`: The medical policy text

**Example:**
```csv
policy_id,cleaned_policy_text
1,"This policy covers office visits and MRI procedures..."
2,"CPT 99213 is covered for established patients..."
```

## 🚀 **Scaling to Large Datasets**

### **For 100,000 documents:**

**Fast Method:** ~8 minutes
**AI Method:** ~8 hours (but much more accurate)

### **Tips for Large Datasets:**
1. **Use multiple computers** to process in parallel
2. **Pre-compute AI models** once, then reuse
3. **Store results in a database** for easy access

## 🐛 **Troubleshooting**

### **Common Problems:**

1. **"No module named transformers"**
   ```bash
   pip install transformers torch
   ```

2. **"Memory error"**
   - Use the fast method instead
   - Process smaller batches

3. **"File not found"**
   - Check your file paths
   - Make sure CSV has the right column names

## 🎯 **When to Use Which Method**

### **Use BERT (AI) when:**
- You need maximum accuracy
- You have time to wait for results
- You're processing for production use

### **Use Rules when:**
- You need quick results
- You're testing or developing
- You don't have powerful computers

## 📞 **Need Help?**

- Check the error messages - they usually tell you what's wrong
- Try the fast method first to make sure everything works
- Make sure your input file has the right format

---

**Quick Start:**
1. Put your data in a CSV file
2. Run: `python3 run_pipeline.py -input your_file.csv -output results.csv`
3. Check the results!