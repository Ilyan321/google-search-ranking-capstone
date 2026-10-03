from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.cell(0, 10, 'FL-01: AI Workflow Audit and Tool Setup', 0, 1, 'C')
        self.set_font('Helvetica', '', 12)
        self.cell(0, 10, 'Intern Name: Ilyan Khan | Submission Date: July 15, 2026', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

pdf = PDF()
pdf.add_page()

# --- Section 1 ---
pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 10, "1. Workflow Audit", 0, 1)
pdf.set_font("Helvetica", "", 10)
pdf.multi_cell(0, 6, "Below is an audit of recurring tasks from my university studies and software development projects, classified by how AI should be integrated into each workflow to maximize efficiency and learning.")
pdf.ln(5)

workflow_data = [
    ["Task", "Category", "Rationale"],
    ["1. Attending live CSE lectures", "Just me", "Requires my physical presence, active listening, and in-the-moment participation."],
    ["2. Defining core logic for new automations (e.g., Slack)", "Just me", "Requires my unique understanding of what specific problem I am trying to solve."],
    ["3. Structuring my daily development schedule", "Just me", "Depends entirely on my personal energy levels and daily university workload."],
    ["4. Cleaning up and organizing raw class notes", "Delegate to AI with review", "AI can format my rough notes into study guides, but I must verify technical accuracy."],
    ["5. Summarizing academic papers/PDFs for university", "Delegate to AI with review", "Speeds up my reading process, but I need to review it to ensure I understand the core concepts."],
    ["6. Writing boilerplate code for new websites/apps", "Delegate to AI with review", "AI generates the foundational code fast, leaving me to just review and tweak the architecture."],
    ["7. Researching documentation for Antigravity agent", "Collaborate with AI", "AI acts as a great interactive tutor to help me understand complex new frameworks faster."],
    ["8. Brainstorming new AI tool architectures", "Collaborate with AI", "Working back-and-forth with an LLM helps me expand on my initial technical ideas."],
    ["9. Debugging scripts (e.g., email reader errors)", "Collaborate with AI", "We can troubleshoot together by pasting error logs and iterating on the solution."],
    ["10. Drafting specific daily queries for Gemini", "Collaborate with AI", "I provide the initial context and intent, and we refine the prompt together to get the best output."],
    ["11. Fetching and reading new emails in Slack", "Fully automate", "Once the automation script is built, it runs entirely on its own via triggers without my input."],
    ["12. Categorizing saved Instagram Reels for inspiration", "Fully automate", "Can be handled by a background webhook (like Zapier) sending saved links straight to a Notion database."]
]

# Set specific column widths (total width usually ~190 in A4)
col_widths = (65, 40, 85)

with pdf.table(borders_layout="ALL", text_align="LEFT", cell_fill_color=200, cell_fill_mode="ROWS", line_height=6, col_widths=col_widths) as table:
    for data_row in workflow_data:
        row = table.row()
        for i, datum in enumerate(data_row):
            if workflow_data.index(data_row) == 0:
                pdf.set_font("Helvetica", "B", 10)
            else:
                pdf.set_font("Helvetica", "", 9)
            row.cell(datum)

pdf.add_page()

# --- Section 2 ---
pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 10, "2. Target Tasks", 0, 1)
pdf.set_font("Helvetica", "", 10)
pdf.multi_cell(0, 6, "The following three tasks have been selected for further refinement in upcoming assignments (FL-02 through FL-04).")
pdf.ln(5)

target_tasks = [
    ["Target Task", "Definition of \"Done Well\""],
    ["Researching documentation for Antigravity agent", "I'll count this as a success if I can spin up a quick technical cheat sheet (including the code snippet, pros, and cons) for a new agent feature in under 60 minutes without getting stuck in infinite loops of reading."],
    ["Creating automations (e.g., Slack email reader)", "Success for me means getting the script running reliably for a full week without it crashing or needing manual fixes, successfully sorting my priority emails exactly how I defined them."],
    ["Summarizing academic papers/PDFs for university", "I'll know I'm doing this well when I can walk away from every paper with three clear bullet points that capture the core methodology, and I can actually explain those points to a classmate from memory."]
]

target_col_widths = (65, 125)

with pdf.table(borders_layout="ALL", text_align="LEFT", cell_fill_color=200, cell_fill_mode="ROWS", line_height=6, col_widths=target_col_widths) as table:
    for data_row in target_tasks:
        row = table.row()
        for i, datum in enumerate(data_row):
            if target_tasks.index(data_row) == 0:
                pdf.set_font("Helvetica", "B", 10)
            else:
                pdf.set_font("Helvetica", "", 10)
            row.cell(datum)

pdf.add_page()

# --- Section 3 ---
pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 10, "3. Visual Evidence", 0, 1)
pdf.set_font("Helvetica", "", 10)
pdf.multi_cell(0, 6, "(Below is the visual evidence verifying my tool workspace configurations and course completion.)")
pdf.ln(5)

images_info = [
    {"title": "Claude Project Configuration", "desc": "Screenshot showing my custom instructions and setup for my dedicated Claude Project.", "file": "pic_1"},
    {"title": "Anthropic Academy Completion", "desc": "Certificate/verification of enrollment and completion of the first module in AI Fluency.", "file": "pic_2"},
    {"title": "ChatGPT Account Verification", "desc": "Screenshot confirming my active ChatGPT workspace configuration.", "file": "pic_3"}
]

for info in images_info:
    # Add a new page if the image + text might run off the bottom
    if pdf.get_y() > 180:
        pdf.add_page()
        
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, info["title"], 0, 1)
    
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 6, info["desc"])
    pdf.ln(2)
    
    if os.path.exists(info["file"]):
        # Read the image extension to help FPDF, since pic_1 etc have no extension
        # If it complains about missing extension, we can rename them or pass a format type if needed.
        try:
            pdf.image(info["file"], w=150)
        except Exception as e:
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 10, f"[Error loading image {info['file']}: {str(e)}]", 0, 1)
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 10, f"[Image file '{info['file']}' not found in directory]", 0, 1)
        
    pdf.ln(10)

pdf.output("Flyrank_Setup_Audit.pdf")
print("Successfully generated Flyrank_Setup_Audit.pdf")
