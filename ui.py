import sys
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTextEdit, QPushButton, QApplication
)

from rag_engine import (
    load_knowledge_base,
    build_index,
    generate_response
)


# =========================================================
# INIT RAG SYSTEM (LOAD + INDEX)
# =========================================================
KB_PATH = "knowledge_base.json"

docs = load_knowledge_base(KB_PATH)
index, docs = build_index(docs)


# =========================================================
# UI CLASS
# =========================================================
class SpyderaUI(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Spydera IDS RAG System")
        self.setGeometry(250, 120, 950, 700)

        # ---------------- UI Layout ----------------
        layout = QVBoxLayout()

        # Input box
        self.input_box = QTextEdit()
        self.input_box.setPlaceholderText("Ask Spydera IDS...")

        # Button
        self.button = QPushButton("Run Query")
        self.button.clicked.connect(self.run_query)

        # Output box
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)

        # Add widgets
        layout.addWidget(self.input_box)
        layout.addWidget(self.button)
        layout.addWidget(self.output_box)

        self.setLayout(layout)

    # =====================================================
    # MAIN ACTION (CALL RAG ENGINE)
    # =====================================================
    def run_query(self):

        query = self.input_box.toPlainText().strip()

        if not query:
            self.output_box.setText("Please enter a query.")
            return

        # Call backend RAG system
        result = generate_response(query, index, docs)

        # Display formatted output
        self.output_box.setText(
            f"""
===============================
SPYDERA IDS RESPONSE
===============================

ANSWER:
{result['answer']}

-------------------------------
CONFIDENCE:
{result['confidence']:.2f}

-------------------------------
SOURCES:
{chr(10).join(result['sources'])}
"""
        )


# =========================================================
# OPTIONAL ENTRY POINT (FOR DIRECT RUN ONLY)
# =========================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpyderaUI()
    window.show()
    sys.exit(app.exec_())