# 🤖 agentic-rag - Smart Document Assistant for Everyone

[![Download agentic-rag](https://img.shields.io/badge/Download-Agentic--RAG-brightgreen?style=for-the-badge)](https://github.com/loloyiy/agentic-rag/releases)

Welcome to **agentic-rag**. This app helps you handle documents smartly. It works with many file types like PDFs, Word, Excel, and CSV files. You can also chat with your documents using Telegram or WhatsApp bots. This guide will help you download, install, and run agentic-rag on your computer step-by-step.

## 🔍 What is agentic-rag?

agentic-rag is a tool that reads your documents and helps you find information fast. It breaks documents into small parts (called chunks) and searches them using smart methods. The app uses both vector search and BM25 search to find the best matches. It supports many document formats, so you can use it for PDFs, Word files, Excel sheets, and more.

You don’t need to know programming to use this app. It has chatbots for popular messaging apps, so you can ask your documents questions directly through Telegram or WhatsApp. The app uses popular technologies like FastAPI for the server, React for the interface, PostgreSQL for storage, and LangChain for managing language models.

## 📥 Download & Install agentic-rag

You can download agentic-rag from the official releases page. This page has the latest versions for Windows, macOS, and Linux.

[![Download agentic-rag](https://img.shields.io/badge/Download-Agentic--RAG-brightgreen?style=for-the-badge)](https://github.com/loloyiy/agentic-rag/releases)

### Step 1: Visit the download page

- Click the button above or go to:  
  https://github.com/loloyiy/agentic-rag/releases  
- Find the version that fits your computer’s operating system (Windows, macOS, or Linux).

### Step 2: Download the installer

- Click the file that matches your OS. These files are usually named with your OS and version number, such as `agentic-rag-windows-setup.exe` or `agentic-rag-macos.dmg`.
- Save the file to your computer.

### Step 3: Install agentic-rag

- Windows: Open the `.exe` file and follow the install wizard.
- macOS: Open the `.dmg` file and drag agentic-rag to your Applications folder.
- Linux: Follow the README instructions inside the downloaded package, usually this involves commands in the terminal like running a `.deb` or `.AppImage` file.

### Step 4: Run the app

- After installation, launch agentic-rag from your Start Menu (Windows), Applications folder (macOS), or however your system runs apps.
- The first time you open it, you might see a welcome screen or tutorial. Follow any instructions there to get started.

## ⚙️ System Requirements

To run agentic-rag smoothly, your computer should meet these basic requirements:

- **Operating System:** Windows 10 or later, macOS 10.15 or later, or a recent Linux distribution (Ubuntu 20.04+ recommended).
- **Processor:** 64-bit, Intel or AMD processor with at least 2 GHz speed.
- **Memory:** Minimum 4 GB of RAM; 8 GB recommended for better performance.
- **Storage:** At least 500 MB of free disk space for the app and space for your documents.
- **Internet:** Required for downloading, signing in if needed, and accessing chatbot services.

## 💡 How to Use agentic-rag

agentic-rag lets you search and chat with your documents easily. Here are common tasks you can do after installing.

### Import Documents

1. Open the app.
2. Find the button or menu to add documents.
3. Select your files (PDF, Word, Excel, CSV).
4. The app will process the documents and prepare them for searching.

### Search Documents

- Use the search bar to type questions or keywords.
- agentic-rag searches through your documents using smart methods to find the best answers.
- Results appear quickly, showing relevant chunks of your documents.

### Use Chatbots

- To chat with your documents on Telegram or WhatsApp:  
  - Follow the instructions inside agentic-rag to link your messaging app.
  - Send your questions to the chatbot.
  - The bot replies with answers based on your documents.

### Manage Documents

- You can remove old documents or add new ones anytime.
- Documents are stored safely in the app’s database.

## 🔧 Troubleshooting

If you encounter issues, try these tips:

- Make sure your internet connection is stable.
- Restart the app.
- Check for updates on the releases page.
- If documents don’t load, try importing them again.
- For chatbot problems, verify the messaging app is connected.

If problems persist, check the app’s support resources or report issues on GitHub.

## 📚 More Information

agentic-rag uses the following core technologies to work efficiently:

- **FastAPI:** A server framework that handles user requests quickly.
- **React:** The app’s user interface technology.
- **PostgreSQL with pgvector:** Stores your documents and supports fast vector searches.
- **LangChain:** Manages how the app talks with language AI models.
- **Hybrid Search:** Combines vector search and classic BM25 search to find better results.
- **Supported Bots:** You can ask your documents questions through Telegram and WhatsApp bots.

This combination ensures agentic-rag provides fast and accurate answers from your documents.

## 🛠️ FAQ

**Q: Do I need to create an account?**  
A: No account is required to use the basic document assistant features.

**Q: Can I add other document types?**  
A: Currently, agentic-rag supports PDFs, Word, Excel, and CSV files.

**Q: Can I use agentic-rag on my phone?**  
A: You can chat with your documents on your phone by linking Telegram or WhatsApp bots. The app itself is designed for desktop computers.

**Q: Is my data safe?**  
A: Your documents are stored locally or on your server using PostgreSQL. Chat data is handled securely.

---

If you want the latest versions or more detailed release notes, visit the official releases page anytime:

https://github.com/loloyiy/agentic-rag/releases