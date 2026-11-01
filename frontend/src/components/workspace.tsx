"use client";

import {
  Activity,
  ArrowUp,
  Atom,
  BookOpenText,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Code2,
  Copy,
  FileCode2,
  FilePlus2,
  FileText,
  FolderKanban,
  Layers3,
  LoaderCircle,
  LogOut,
  Menu,
  MessageSquareText,
  Moon,
  MoreHorizontal,
  Pencil,
  Plus,
  Search,
  Send,
  Settings2,
  Sparkles,
  Sun,
  Trash2,
  X,
} from "lucide-react";
import React, { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown, { Components } from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import {
  api,
  Conversation,
  DocumentFile,
  Memory,
  Message,
  BuildTarget,
  Mode,
  ModelCatalog,
  Project,
  Source,
  User,
  streamPost,
} from "@/lib/api";

const MODE_INFO: { id: Mode; name: string; summary: string; icon: typeof Code2 }[] = [
  { id: "code", name: "Code", summary: "Write, debug and understand software.", icon: Code2 },
  { id: "research", name: "Research", summary: "Explore papers and technical sources.", icon: BookOpenText },
  { id: "architecture", name: "Architecture", summary: "Shape systems, APIs and diagrams.", icon: Layers3 },
  { id: "engineering", name: "Engineering", summary: "Work through maths and technical models.", icon: Atom },
];

const MARKDOWN_COMPONENTS: Components = {
  a: ({ href, children, ...props }) => <a href={href} target="_blank" rel="noreferrer" {...props}>{children}</a>,
  pre: ({ children, ...props }) => <pre className="code-frame" {...props}>{children}</pre>,
};

function formatDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(date);
}

function shortSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function Workspace() {
  const [user, setUser] = useState<User | null>(null);
  const [checkingSession, setCheckingSession] = useState(true);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authError, setAuthError] = useState("");
  const [authBusy, setAuthBusy] = useState(false);
  const [projects, setProjects] = useState<Project[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [mode, setMode] = useState<Mode>("code");
  const [buildTarget, setBuildTarget] = useState<BuildTarget>("auto");
  const [catalog, setCatalog] = useState<ModelCatalog>({ provider: "", configured: false, models: [] });
  const [modelName, setModelName] = useState("");
  const [prompt, setPrompt] = useState("");
  const [webSearch, setWebSearch] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamError, setStreamError] = useState("");
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);
  const [searchText, setSearchText] = useState("");
  const [documents, setDocuments] = useState<DocumentFile[]>([]);
  const [memories, setMemories] = useState<Memory[]>([]);
  const [memoryDraft, setMemoryDraft] = useState("");
  const [memoryLabel, setMemoryLabel] = useState("");
  const [projectDialog, setProjectDialog] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [projectDescription, setProjectDescription] = useState("");
  const [projectInstructions, setProjectInstructions] = useState("");
  const [rightOpen, setRightOpen] = useState(true);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [authNotice, setAuthNotice] = useState("");
  const [sidebarError, setSidebarError] = useState("");
  const streamAbort = useRef<AbortController | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const activeProject = projects.find((project) => project.id === activeProjectId) || null;
  const activeConversation = conversations.find((item) => item.id === activeConversationId) || null;
  const visibleConversations = useMemo(() => {
    const query = searchText.toLowerCase().trim();
    return conversations.filter((item) => {
      const belongsToProject = !activeProjectId || item.project_id === activeProjectId;
      return belongsToProject && (!query || item.title.toLowerCase().includes(query));
    });
  }, [conversations, searchText, activeProjectId]);

  const refreshWorkspace = useCallback(async () => {
    try {
      const [projectList, conversationList, modelCatalog, memoryList] = await Promise.all([
        api<Project[]>("/api/projects"),
        api<Conversation[]>("/api/conversations"),
        api<ModelCatalog>("/api/models"),
        api<Memory[]>("/api/memories"),
      ]);
      setProjects(projectList);
      setConversations(conversationList);
      setCatalog(modelCatalog);
      setMemories(memoryList);
      setModelName((current) => current || modelCatalog.models[0]?.id || "");
    } catch (error) {
      setSidebarError(error instanceof Error ? error.message : "Could not load your workspace.");
    }
  }, []);

  useEffect(() => {
    const storedTheme = window.localStorage.getItem("medha-theme");
    if (storedTheme === "light" || storedTheme === "dark") {
      window.requestAnimationFrame(() => setTheme(storedTheme));
    }
    api<User>("/api/auth/me")
      .then((currentUser) => {
        setUser(currentUser);
        return refreshWorkspace();
      })
      .catch(() => setUser(null))
      .finally(() => setCheckingSession(false));
  }, [refreshWorkspace]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("medha-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!activeProjectId) {
      return;
    }
    let active = true;
    Promise.all([
      api<DocumentFile[]>(`/api/projects/${activeProjectId}/documents`),
      api<Memory[]>("/api/memories"),
    ]).then(([files, memoryList]) => {
      if (!active) return;
      setDocuments(files);
      setMemories(memoryList);
    }).catch((error) => {
      if (active) setSidebarError(error instanceof Error ? error.message : "Could not load project context.");
    });
    return () => { active = false; };
  }, [activeProjectId]);

  const openConversation = async (conversation: Conversation) => {
    setActiveConversationId(conversation.id);
    setActiveProjectId(conversation.project_id);
    setDocuments([]);
    setMode(conversation.mode);
    setModelName(conversation.model_name || catalog.models[0]?.id || "");
    setMobileSidebarOpen(false);
    setStreamError("");
    setIsLoadingMessages(true);
    try {
      setMessages(await api<Message[]>(`/api/conversations/${conversation.id}/messages`));
    } catch (error) {
      setStreamError(error instanceof Error ? error.message : "Could not load this conversation.");
    } finally {
      setIsLoadingMessages(false);
    }
  };

  const startNewChat = () => {
    setActiveConversationId(null);
    setMessages([]);
    setMode("code");
    setPrompt("");
    setStreamError("");
    setMobileSidebarOpen(false);
    window.setTimeout(() => inputRef.current?.focus(), 10);
  };

  const runStream = async (
    path: string,
    body: unknown,
    conversationId: string,
    temporaryId: string,
  ) => {
    const controller = new AbortController();
    streamAbort.current = controller;
    setIsStreaming(true);
    setStreamError("");
    try {
      await streamPost(path, body, {
        onDelta: (delta) => setMessages((current) => current.map((message) => message.id === temporaryId ? { ...message, content: message.content + delta } : message)),
        onDone: (event) => {
          const sources = Array.isArray(event.sources) ? event.sources as Source[] : [];
          setMessages((current) => current.map((message) => message.id === temporaryId ? { ...message, sources, model_name: String(event.model || modelName) } : message));
        },
        onError: (message) => {
          setStreamError(message);
          setMessages((current) => current.map((item) => item.id === temporaryId ? { ...item, content: item.content || `Medha could not finish: ${message}` } : item));
        },
      }, controller.signal);
      const savedMessages = await api<Message[]>(`/api/conversations/${conversationId}/messages`);
      setMessages(savedMessages);
      const savedConversations = await api<Conversation[]>("/api/conversations");
      setConversations(savedConversations);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        setStreamError("Generation stopped.");
      } else {
        const message = error instanceof Error ? error.message : "The model request failed.";
        setStreamError(message);
        setMessages((current) => current.map((item) => item.id === temporaryId ? { ...item, content: item.content || `Medha is not ready: ${message}` } : item));
      }
    } finally {
      streamAbort.current = null;
      setIsStreaming(false);
      inputRef.current?.focus();
    }
  };

  const sendMessage = async (event?: FormEvent) => {
    event?.preventDefault();
    const content = prompt.trim();
    if (!content || isStreaming) return;
    setPrompt("");
    setStreamError("");
    let conversationId = activeConversationId;
    try {
      if (!conversationId) {
        const conversation = await api<Conversation>("/api/conversations", {
          method: "POST",
          body: JSON.stringify({ title: "New conversation", project_id: activeProjectId, mode, model_name: modelName || null }),
        });
        conversationId = conversation.id;
        setActiveConversationId(conversation.id);
        setConversations((current) => [conversation, ...current]);
      }
      const now = new Date().toISOString();
      const assistantId = `stream-${crypto.randomUUID()}`;
      setMessages((current) => [
        ...current,
        { id: `local-${crypto.randomUUID()}`, role: "user", content, model_name: modelName, input_tokens: null, output_tokens: null, sources: [], created_at: now },
        { id: assistantId, role: "assistant", content: "", model_name: modelName, input_tokens: null, output_tokens: null, sources: [], created_at: now },
      ]);
      await runStream(`/api/conversations/${conversationId}/messages`, { content, mode, model_name: modelName || null, web_search: webSearch, build_target: buildTarget }, conversationId, assistantId);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not create the conversation.";
      setStreamError(message);
      setPrompt(content);
    }
  };

  const regenerate = async (message: Message) => {
    if (!activeConversationId || isStreaming) return;
    setMessages((current) => current.map((item) => item.id === message.id ? { ...item, content: "", sources: [] } : item));
    await runStream(`/api/conversations/${activeConversationId}/messages/${message.id}/regenerate`, {}, activeConversationId, message.id);
  };

  const resendEdited = async (message: Message, content: string) => {
    if (!activeConversationId || isStreaming) return;
    try {
      await api<Message>(`/api/conversations/${activeConversationId}/messages/${message.id}`, { method: "PATCH", body: JSON.stringify({ content }) });
      setMessages((current) => current.map((item) => item.id === message.id ? { ...item, content } : item));
      const assistantId = `stream-${crypto.randomUUID()}`;
      setMessages((current) => [...current.filter((item) => item.created_at <= message.created_at), { id: assistantId, role: "assistant", content: "", model_name: modelName, input_tokens: null, output_tokens: null, sources: [], created_at: new Date().toISOString() }]);
      await runStream(`/api/conversations/${activeConversationId}/messages/${message.id}/resend`, { web_search: webSearch, model_name: modelName || null, mode, build_target: buildTarget }, activeConversationId, assistantId);
    } catch (error) {
      setStreamError(error instanceof Error ? error.message : "Could not resend the edited message.");
    }
  };

  const createProject = async (event: FormEvent) => {
    event.preventDefault();
    if (!projectName.trim()) return;
    try {
      const project = await api<Project>("/api/projects", {
        method: "POST",
        body: JSON.stringify({ name: projectName.trim(), description: projectDescription, instructions: projectInstructions }),
      });
      setProjects((current) => [project, ...current]);
      setActiveProjectId(project.id);
      setDocuments([]);
      setProjectDialog(false);
      setProjectName("");
      setProjectDescription("");
      setProjectInstructions("");
      setSidebarError("");
    } catch (error) {
      setSidebarError(error instanceof Error ? error.message : "Could not create the project.");
    }
  };

  const uploadFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file || !activeProjectId) return;
    const formData = new FormData();
    formData.set("file", file);
    try {
      const uploaded = await api<DocumentFile>(`/api/projects/${activeProjectId}/documents`, { method: "POST", body: formData });
      setDocuments((current) => [uploaded, ...current]);
      setSidebarError("");
    } catch (error) {
      setSidebarError(error instanceof Error ? error.message : "File upload failed.");
    }
  };

  const removeDocument = async (documentId: string) => {
    if (!activeProjectId) return;
    try {
      await api(`/api/projects/${activeProjectId}/documents/${documentId}`, { method: "DELETE" });
      setDocuments((current) => current.filter((item) => item.id !== documentId));
    } catch (error) {
      setSidebarError(error instanceof Error ? error.message : "Could not delete this file.");
    }
  };

  const addMemory = async (event: FormEvent) => {
    event.preventDefault();
    if (!memoryLabel.trim() || !memoryDraft.trim()) return;
    try {
      const memory = await api<Memory>("/api/memories", {
        method: "POST",
        body: JSON.stringify({ label: memoryLabel.trim(), content: memoryDraft.trim(), project_id: activeProjectId }),
      });
      setMemories((current) => [memory, ...current]);
      setMemoryDraft("");
      setMemoryLabel("");
    } catch (error) {
      setSidebarError(error instanceof Error ? error.message : "Could not save that memory.");
    }
  };

  const deleteMemory = async (memoryId: string) => {
    try {
      await api(`/api/memories/${memoryId}`, { method: "DELETE" });
      setMemories((current) => current.filter((item) => item.id !== memoryId));
    } catch (error) {
      setSidebarError(error instanceof Error ? error.message : "Could not remove that memory.");
    }
  };

  const deleteConversation = async (conversation: Conversation) => {
    try {
      await api(`/api/conversations/${conversation.id}`, { method: "DELETE" });
      setConversations((current) => current.filter((item) => item.id !== conversation.id));
      if (activeConversationId === conversation.id) startNewChat();
    } catch (error) {
      setSidebarError(error instanceof Error ? error.message : "Could not delete this conversation.");
    }
  };

  const logout = async () => {
    await api("/api/auth/logout", { method: "POST" }).catch(() => undefined);
    setUser(null);
    setConversations([]);
    setProjects([]);
    setMessages([]);
  };

  const submitAuth = async (event: FormEvent) => {
    event.preventDefault();
    setAuthBusy(true);
    setAuthError("");
    try {
      const currentUser = await api<User>(`/api/auth/${authMode === "register" ? "register" : "login"}`, {
        method: "POST",
        body: JSON.stringify({ email: authEmail, password: authPassword }),
      });
      setUser(currentUser);
      setAuthPassword("");
      await refreshWorkspace();
    } catch (error) {
      setAuthError(error instanceof Error ? error.message : "Sign-in failed.");
    } finally {
      setAuthBusy(false);
    }
  };

  if (checkingSession) return <div className="boot-screen"><span className="brand-mark">M</span><LoaderCircle className="spin" size={18} /></div>;

  if (!user) {
    return (
      <main className="auth-screen">
        <section className="auth-card">
          <div className="auth-brand"><span className="brand-mark">M</span><span>medha<span className="brand-period">.</span></span></div>
          <div className="auth-kicker"><Activity size={14} /> TECHNICAL INTELLIGENCE WORKSPACE</div>
          <h1>{authMode === "register" ? "Create your workspace" : "Welcome back"}</h1>
          <p className="auth-copy">Think through code, research and systems in one focused place.</p>
          <form className="auth-form" onSubmit={submitAuth}>
            <label>Email address<input type="email" autoComplete="email" value={authEmail} onChange={(event) => setAuthEmail(event.target.value)} required /></label>
            <label>Password<input type="password" minLength={authMode === "register" ? 10 : 1} autoComplete={authMode === "register" ? "new-password" : "current-password"} value={authPassword} onChange={(event) => setAuthPassword(event.target.value)} required /></label>
            {authMode === "register" && <span className="field-hint">Use at least 10 characters.</span>}
            {authError && <div className="inline-error" role="alert">{authError}</div>}
            <button className="primary-button auth-submit" disabled={authBusy}>{authBusy ? <LoaderCircle size={16} className="spin" /> : null}{authMode === "register" ? "Create account" : "Sign in"}<ArrowUp size={15} /></button>
          </form>
          <p className="auth-switch">{authMode === "register" ? "Already have a workspace?" : "New to Medha?"} <button onClick={() => { setAuthMode(authMode === "register" ? "login" : "register"); setAuthError(""); }}>{authMode === "register" ? "Sign in" : "Create an account"}</button></p>
          <div className="auth-security"><span className="security-dot" /> Personal projects are private to your account.</div>
        </section>
      </main>
    );
  }

  const currentMode = MODE_INFO.find((item) => item.id === mode) || MODE_INFO[0];
  const ModeIcon = currentMode.icon;

  return (
    <main className="workspace-shell" data-sidebar-open={mobileSidebarOpen}>
      {mobileSidebarOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMobileSidebarOpen(false)} />}
      <aside className="sidebar">
        <div className="sidebar-head">
          <button className="wordmark" onClick={startNewChat} aria-label="Medha home"><span className="brand-mark">M</span><span>medha<span className="brand-period">.</span></span><span className="version-tag">LAB</span></button>
          <button className="icon-button sidebar-close" onClick={() => setMobileSidebarOpen(false)} aria-label="Close sidebar"><X size={17} /></button>
        </div>
        <button className="new-chat-button" onClick={startNewChat}><Plus size={16} /><span>New conversation</span><kbd>⌘ K</kbd></button>
        <label className="sidebar-search"><Search size={15} /><input value={searchText} onChange={(event) => setSearchText(event.target.value)} placeholder="Search conversations" aria-label="Search conversations" /><kbd>/</kbd></label>
        <div className="sidebar-section-head"><span>MODES</span></div>
        <nav className="mode-nav" aria-label="Technical modes">
          {MODE_INFO.map((item) => {
            const Icon = item.icon;
            return <button key={item.id} className={`mode-nav-item ${mode === item.id ? "active" : ""}`} onClick={() => setMode(item.id)}><Icon size={16} /><span>{item.name}</span>{mode === item.id && <span className="nav-current" />}</button>;
          })}
        </nav>
        <div className="sidebar-section-head projects-head"><span>PROJECTS</span><button onClick={() => setProjectDialog(true)} aria-label="Create project"><Plus size={15} /></button></div>
        <div className="project-list">
          <button className={`project-row ${activeProjectId === null ? "selected" : ""}`} onClick={() => { setActiveProjectId(null); setDocuments([]); }}><FolderKanban size={15} /><span>All conversations</span></button>
          {projects.map((project) => <button key={project.id} className={`project-row ${activeProjectId === project.id ? "selected" : ""}`} onClick={() => { setActiveProjectId(project.id); setDocuments([]); setActiveConversationId(null); setMessages([]); }}><span className="project-dot" /> <span>{project.name}</span><span className="project-count">{conversations.filter((item) => item.project_id === project.id).length}</span></button>)}
          {!projects.length && <p className="sidebar-empty">Create a project to keep files and conversations together.</p>}
        </div>
        <div className="sidebar-section-head recent-head"><span>RECENT</span><button aria-label="Refresh conversations" onClick={() => refreshWorkspace()}><MoreHorizontal size={16} /></button></div>
        <div className="conversation-list">
          {visibleConversations.map((conversation) => <div className={`conversation-row ${activeConversationId === conversation.id ? "selected" : ""}`} key={conversation.id}>
            <button onClick={() => void openConversation(conversation)}><MessageSquareText size={14} /><span>{conversation.title}</span></button>
            <button className="row-delete" aria-label={`Delete ${conversation.title}`} onClick={() => void deleteConversation(conversation)}><Trash2 size={13} /></button>
          </div>)}
          {!visibleConversations.length && <p className="sidebar-empty">{searchText ? "No conversations match your search." : "Your recent work will appear here."}</p>}
        </div>
        {sidebarError && <p className="sidebar-error" role="status">{sidebarError}</p>}
        <div className="sidebar-footer">
          <button className="profile-row" onClick={() => setAuthNotice("Your account is stored in this Medha deployment.")}><span className="profile-avatar">{user.email.slice(0, 1).toUpperCase()}</span><span className="profile-details"><strong>{user.email.split("@")[0]}</strong><small>{user.email}</small></span><ChevronDown size={14} /></button>
          <div className="footer-tools"><button className="icon-button" title="Settings" onClick={() => setAuthNotice(`AI provider: ${catalog.provider || "not configured"}. Set provider details in the server environment.`)}><Settings2 size={16} /></button><button className="icon-button" title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"} onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}</button><button className="icon-button" title="Sign out" onClick={() => void logout()}><LogOut size={16} /></button></div>
        </div>
      </aside>

      <section className="center-column">
        <header className="topbar">
          <div className="topbar-left"><button className="icon-button mobile-menu" onClick={() => setMobileSidebarOpen(true)} aria-label="Open navigation"><Menu size={18} /></button><div className="mode-heading"><span className={`mode-icon mode-${mode}`}><ModeIcon size={15} /></span><div><div className="mode-title">Medha {currentMode.name}</div><div className="mode-subtitle">{activeProject ? activeProject.name : "Technical intelligence workspace"}</div></div></div></div>
          <div className="topbar-right"><span className={`provider-indicator ${catalog.configured ? "ready" : ""}`} title={catalog.configured ? `Connected to ${catalog.provider}` : "Configure an AI provider on the server"}><span />{catalog.configured ? catalog.provider : "Model setup needed"}</span><button className="icon-button" onClick={() => setRightOpen(!rightOpen)} title={rightOpen ? "Hide context panel" : "Show context panel"} aria-label="Toggle context panel">{rightOpen ? <ChevronRight size={17} /> : <ChevronLeft size={17} />}</button></div>
        </header>

        <div className="chat-scroll" id="chat-scroll">
          <div className="chat-inner">
            {!activeConversationId && messages.length === 0 && (
              <div className="welcome-panel">
                <div className="welcome-orbit"><div className="welcome-core"><Sparkles size={22} /></div><span className="orbit orbit-one" /><span className="orbit orbit-two" /><span className="orbit orbit-three" /></div>
                <p className="eyebrow">A CLEARER WAY THROUGH COMPLEX WORK</p>
                <h1>What are you<br /><span>working through?</span></h1>
                <p className="welcome-copy">Move from the first question to a working technical solution, with your projects and research in context.</p>
                <div className="mode-cards">
                  {MODE_INFO.map((item) => { const Icon = item.icon; return <button className={`mode-card ${mode === item.id ? "picked" : ""}`} key={item.id} onClick={() => setMode(item.id)}><span><Icon size={17} /></span><strong>{item.name}</strong><small>{item.summary}</small></button>; })}
                </div>
                <div className="starter-prompts"><span>TRY</span><button onClick={() => setPrompt("Explain how a transformer uses self-attention, with a small diagram.")}>Explain self-attention <ArrowUp size={12} /></button><button onClick={() => { setMode("architecture"); setPrompt("Design a reliable notification service for a multi-tenant SaaS product. Include a Mermaid diagram."); }}>Sketch a system design <ArrowUp size={12} /></button><button onClick={() => { setMode("research"); setPrompt("Find and compare recent public research on retrieval-augmented generation."); }}>Explore technical research <ArrowUp size={12} /></button></div>
              </div>
            )}
            {isLoadingMessages && <div className="loading-thread"><LoaderCircle className="spin" size={18} /> Loading conversation…</div>}
            <div className="message-stack">
              {messages.map((message) => <MessageRow key={message.id} message={message} streaming={isStreaming && message.id.startsWith("stream-")} disabled={isStreaming} onRegenerate={() => void regenerate(message)} onResend={(content) => void resendEdited(message, content)} />)}
            </div>
            {streamError && <div className="stream-status" role="status"><span />{streamError}</div>}
            {authNotice && <div className="notice-banner"><span>{authNotice}</span><button onClick={() => setAuthNotice("")} aria-label="Dismiss"><X size={14} /></button></div>}
            <div className="bottom-space" />
          </div>
        </div>

        <div className="composer-area">
          {!catalog.configured && <div className="provider-callout"><Settings2 size={14} /><span>Connect an AI model provider in the server environment to start a real conversation.</span></div>}
          <form className="composer" onSubmit={(event) => void sendMessage(event)}>
            <textarea ref={inputRef} value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void sendMessage(); } }} placeholder={`Ask Medha ${currentMode.name.toLowerCase()}…`} rows={1} aria-label="Message Medha" />
            <div className="composer-bottom">
              <div className="composer-tools">
                <button className="tool-button" type="button" onClick={() => activeProjectId ? fileInput.current?.click() : setAuthNotice("Select or create a project before adding files.")} title={activeProjectId ? "Attach a project file" : "Choose a project before attaching files"}><FilePlus2 size={15} /><span>Attach</span></button>
                <input ref={fileInput} type="file" className="visually-hidden" accept=".pdf,.txt,.md,.py,.js,.ts,.tsx,.jsx,.html,.css,.json,.csv" onChange={(event) => void uploadFile(event)} />
                <label className="web-toggle"><input type="checkbox" checked={webSearch} onChange={(event) => setWebSearch(event.target.checked)} /><span className="toggle-track"><span /></span><span>Web sources</span></label>
                <span className="composer-divider" />
                <select aria-label="Technical mode" value={mode} onChange={(event) => setMode(event.target.value as Mode)}>
                  {MODE_INFO.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
                </select>
                <select aria-label="App building target" value={buildTarget} onChange={(event) => setBuildTarget(event.target.value as BuildTarget)}>
                  <option value="auto">App target: Auto</option><option value="apple">iPhone · iPad · Mac</option><option value="android">Android</option><option value="windows">Windows</option><option value="cross_platform">All platforms</option>
                </select>
                <select aria-label="AI model" value={modelName} onChange={(event) => setModelName(event.target.value)} disabled={!catalog.models.length}>
                  {!catalog.models.length && <option value="">No model configured</option>}
                  {catalog.models.map((model) => <option key={model.id} value={model.id}>{model.name}</option>)}
                </select>
              </div>
              {isStreaming ? <button className="send-button stop-button" type="button" onClick={() => streamAbort.current?.abort()} aria-label="Stop generation"><X size={17} /></button> : <button className="send-button" type="submit" disabled={!prompt.trim()} aria-label="Send message"><ArrowUp size={18} /></button>}
            </div>
          </form>
          <div className="composer-footnote">Medha can be wrong. Check generated code, citations and calculations before relying on them.{webSearch && <span> · Public search receives your question.</span>}</div>
        </div>
      </section>

      {rightOpen && <aside className="context-panel">
        <div className="context-head"><div><span className="eyebrow">WORKSPACE CONTEXT</span><h2>{activeProject ? activeProject.name : "Project context"}</h2></div><button className="icon-button context-close" onClick={() => setRightOpen(false)} aria-label="Close context panel"><X size={16} /></button></div>
        <section className="context-section">
          <div className="context-section-head"><div><span>PROJECT FILES</span><small>{documents.length} indexed</small></div><button className="small-icon-button" onClick={() => activeProjectId ? fileInput.current?.click() : setProjectDialog(true)} aria-label={activeProjectId ? "Add project file" : "Create a project to add files"}><Plus size={15} /></button></div>
          {!activeProject && <div className="context-empty"><FileCode2 size={18} /><p>Choose a project to keep files, conversations and instructions together.</p><button onClick={() => setProjectDialog(true)}>Create project</button></div>}
          {activeProject && documents.length === 0 && <div className="context-empty"><FileText size={18} /><p>Upload code or research files for Medha to use as project context.</p><button onClick={() => fileInput.current?.click()}>Add a file</button></div>}
          <div className="document-list">{documents.map((file) => <div className="document-row" key={file.id}><span className="file-type"><FileText size={15} /></span><span className="document-meta"><strong title={file.filename}>{file.filename}</strong><small>{shortSize(file.size_bytes)} · {file.chunk_count} passages</small></span><button onClick={() => void removeDocument(file.id)} className="small-icon-button" title={`Remove ${file.filename}`}><Trash2 size={14} /></button></div>)}</div>
        </section>
        <section className="context-section memory-section">
          <div className="context-section-head"><div><span>MEMORY</span><small>{memories.length} saved</small></div></div>
          <form className="memory-form" onSubmit={(event) => void addMemory(event)}><input value={memoryLabel} onChange={(event) => setMemoryLabel(event.target.value)} placeholder="Label" aria-label="Memory label" maxLength={120} /><textarea value={memoryDraft} onChange={(event) => setMemoryDraft(event.target.value)} placeholder="A preference or project fact to remember…" aria-label="Memory content" rows={3} maxLength={4000} /><button disabled={!memoryLabel.trim() || !memoryDraft.trim()}><Plus size={13} />Save memory</button></form>
          <div className="memory-list">{memories.filter((item) => !activeProjectId || !item.project_id || item.project_id === activeProjectId).slice(0, 5).map((item) => <div className="memory-row" key={item.id}><div><strong>{item.label}</strong><p>{item.content}</p></div><button className="small-icon-button" onClick={() => void deleteMemory(item.id)} aria-label={`Delete memory ${item.label}`}><X size={13} /></button></div>)}{!memories.length && <p className="context-footnote">Saved preferences stay under your control.</p>}</div>
        </section>
        {activeConversation && <section className="context-section reference-section"><div className="context-section-head"><div><span>REFERENCES</span><small>{messages.reduce((count, message) => count + message.sources.length, 0)} cited</small></div></div><div className="reference-list">{messages.flatMap((message) => message.sources).slice(-8).reverse().map((source, index) => <SourceCard key={`${source.url}-${index}`} source={source} compact />)}{!messages.some((message) => message.sources.length) && <p className="context-footnote">References from this conversation appear here.</p>}</div></section>}
        <div className="context-footer"><span className="privacy-mark"><span />PRIVATE WORKSPACE</span><span>Personal account</span></div>
      </aside>}

      {projectDialog && <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setProjectDialog(false); }}><section className="modal-card" role="dialog" aria-modal="true" aria-labelledby="project-dialog-title"><div className="modal-head"><div><span className="eyebrow">PROJECT WORKSPACE</span><h2 id="project-dialog-title">Create a project</h2></div><button className="icon-button" onClick={() => setProjectDialog(false)} aria-label="Close"><X size={17} /></button></div><form className="project-form" onSubmit={(event) => void createProject(event)}><label>Project name<input autoFocus value={projectName} onChange={(event) => setProjectName(event.target.value)} placeholder="e.g. Medha Development" maxLength={120} required /></label><label>Description<textarea value={projectDescription} onChange={(event) => setProjectDescription(event.target.value)} placeholder="What is this project about?" rows={3} maxLength={4000} /></label><label>Project instructions<textarea value={projectInstructions} onChange={(event) => setProjectInstructions(event.target.value)} placeholder="Tell Medha how to approach work in this project." rows={3} maxLength={8000} /></label><div className="modal-actions"><button type="button" className="secondary-button" onClick={() => setProjectDialog(false)}>Cancel</button><button className="primary-button" disabled={!projectName.trim()}><Plus size={15} />Create project</button></div></form></section></div>}
    </main>
  );
}

function MessageRow({
  message,
  streaming,
  disabled,
  onRegenerate,
  onResend,
}: {
  message: Message;
  streaming: boolean;
  disabled: boolean;
  onRegenerate: () => void;
  onResend: (content: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(message.content);
  const [copied, setCopied] = useState(false);
  const isUser = message.role === "user";

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  };

  return (
    <article className={`message-row ${isUser ? "from-user" : "from-assistant"}`}>
      <div className={`message-avatar ${isUser ? "user-avatar" : "assistant-avatar"}`}>{isUser ? "P" : <Sparkles size={15} />}</div>
      <div className="message-main">
        <div className="message-meta"><strong>{isUser ? "You" : "Medha"}</strong>{!isUser && message.model_name && <span>{message.model_name}</span>}<time>{formatDate(message.created_at)}</time>{streaming && <span className="live-label"><span />GENERATING</span>}</div>
        {editing ? <form className="edit-message-form" onSubmit={(event) => { event.preventDefault(); onResend(draft); setEditing(false); }}><textarea value={draft} onChange={(event) => setDraft(event.target.value)} rows={Math.min(10, Math.max(3, draft.split("\n").length))} /><div><button type="button" className="secondary-button" onClick={() => { setEditing(false); setDraft(message.content); }}>Cancel</button><button className="primary-button" disabled={!draft.trim()}><Send size={13} />Save & resend</button></div></form> : <div className={`message-content ${isUser ? "user-content" : "assistant-content"}`}>
          {message.content ? <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeHighlight, rehypeKatex]} components={MARKDOWN_COMPONENTS}>{message.content}</ReactMarkdown> : streaming ? <span className="typing-cursor" aria-label="Generating" /> : <span className="muted-response">No response was saved.</span>}
          {streaming && message.content && <span className="typing-cursor" />}
        </div>}
        {!isUser && message.sources?.length > 0 && <div className="source-grid">{message.sources.map((source, index) => <SourceCard key={`${source.url}-${index}`} source={source} />)}</div>}
        {!editing && <div className="message-actions">
          <button onClick={() => void copy()} title="Copy message">{copied ? <Check size={13} /> : <Copy size={13} />}<span>{copied ? "Copied" : "Copy"}</span></button>
          {isUser && <button onClick={() => { setDraft(message.content); setEditing(true); }} disabled={disabled} title="Edit and resend"><Pencil size={13} /><span>Edit</span></button>}
          {!isUser && !streaming && <button onClick={onRegenerate} disabled={disabled} title="Regenerate response"><Activity size={13} /><span>Regenerate</span></button>}
        </div>}
      </div>
    </article>
  );
}

function SourceCard({ source, compact = false }: { source: Source; compact?: boolean }) {
  const external = source.url.startsWith("https://") || source.url.startsWith("http://");
  return <div className={`source-card ${compact ? "compact" : ""}`}>
    <span className="source-kind">{source.kind || "REFERENCE"}</span>
    {external ? <a href={source.url} target="_blank" rel="noreferrer">{source.title}</a> : <strong>{source.title}</strong>}
    {(source.authors || source.year || source.venue) && <small>{[source.authors, source.year, source.venue].filter(Boolean).join(" · ")}</small>}
    {source.snippet && <p>{source.snippet}</p>}
    {source.attribution && <small>{source.attribution}</small>}
  </div>;
}
