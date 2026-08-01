import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { listTopics } from '../registry/topicRegistry';

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    const onOpen = () => { setOpen(true); };
    document.addEventListener('mlv:open-palette', onOpen);
    return () => document.removeEventListener('mlv:open-palette', onOpen);
  }, []);

  if (!open) return null;

  const results = listTopics().filter((t) =>
    t.title.toLowerCase().includes(q.toLowerCase())
  );

  return (
    <div className="palette-overlay" onClick={() => setOpen(false)}>
      <div className="palette" onClick={(e) => e.stopPropagation()}>
        <input autoFocus placeholder="Search topics, concepts, formulas… (S)"
          value={q} onChange={(e) => setQ(e.target.value)} />
        <div className="palette-results">
          {results.map((t) => (
            <button key={t.id} onClick={() => { navigate(`/topic/${t.id}`); setOpen(false); }}>
              {t.title}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
