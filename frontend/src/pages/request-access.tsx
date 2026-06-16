import { useState } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import Head from 'next/head';
import {
  createUserWithEmailAndPassword,
  updateProfile,
} from 'firebase/auth';
import { doc, setDoc, serverTimestamp } from 'firebase/firestore';
import axios from 'axios';
import { auth, db } from '../lib/firebase';

const REFERRAL_OPTIONS = [
  'Select one...',
  'Google search',
  'Friend or colleague',
  'Social media',
  'Sports podcast or blog',
  'Email / newsletter',
  'Reddit',
  'X (Twitter)',
  'Other',
];

export default function RequestAccess() {
  const router = useRouter();
  const [form, setForm] = useState<Record<string, string>>({});
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [done, setDone] = useState(false);

  const set = (key: string, val: string) => setForm(f => ({ ...f, [key]: val }));

  const handleSubmit = async () => {
    setError('');

    if (!form.firstName?.trim()) { setError('First name is required.'); return; }
    if (!form.lastName?.trim()) { setError('Last name is required.'); return; }
    if (!form.email?.trim()) { setError('Email address is required.'); return; }
    if (!form.password?.trim()) { setError('Password is required.'); return; }
    if (form.password.length < 8) { setError('Password must be at least 8 characters.'); return; }

    setSubmitting(true);

    try {
      // Create Firebase Auth user — immediately disabled until approved
      const cred = await createUserWithEmailAndPassword(auth, form.email, form.password);
      const uid = cred.user.uid;

      await updateProfile(cred.user, {
        displayName: `${form.firstName} ${form.lastName}`,
      });

      // Sign out immediately before writing Firestore
      // (security: unapproved users should never have an active session)
      await auth.signOut();

      // Save to Firestore with status: pending
      await setDoc(doc(db, 'users', uid), {
        uid,
        firstName: form.firstName,
        lastName: form.lastName,
        email: form.email,
        phone: form.phone || '',
        referral: form.referral || '',
        status: 'pending',
        createdAt: serverTimestamp(),
      });

      // Notify admin
      await axios.post('/api/notify-request', {
        uid,
        firstName: form.firstName,
        lastName: form.lastName,
        email: form.email,
        phone: form.phone || '',
        referral: form.referral || '',
      });

      // Disable the account server-side via admin SDK
      await axios.post('/api/disable-user', { uid });

      setDone(true);
    } catch (e: any) {
      const code = e?.code;
      if (code === 'auth/email-already-in-use') {
        setError('That email is already registered. Try logging in.');
      } else if (code === 'auth/weak-password') {
        setError('Password must be at least 6 characters.');
      } else if (code === 'auth/invalid-email') {
        setError('Enter a valid email address.');
      } else {
        setError(e.message || 'Something went wrong. Try again.');
      }
    } finally {
      setSubmitting(false);
    }
  };

  if (done) {
    return (
      <>
        <Head><title>Request Submitted — Prime Picks</title></Head>
        <div className="field-bg min-h-screen flex items-center justify-center px-4">
          <div className="panel rounded-xl p-8 max-w-sm w-full text-center animate-fade-up">
            <div style={{ fontSize: 48, marginBottom: 16 }}>✅</div>
            <h2 className="score-display text-chalk mb-3" style={{ fontSize: 26, letterSpacing: '0.08em' }}>
              REQUEST SUBMITTED
            </h2>
            <p className="text-sm mb-2" style={{ color: '#8B9BB4', fontFamily: 'var(--font-mono)', lineHeight: 1.6 }}>
              Thanks, {form.firstName}. Your access request is under review.
            </p>
            <p className="text-xs mb-6" style={{ color: '#4A5568', fontFamily: 'var(--font-mono)' }}>
              You'll receive a confirmation at{' '}
              <span style={{ color: '#C9A84C' }}>{form.email}</span>{' '}
              once approved.
            </p>
            <Link
              href="/login"
              className="text-xs underline"
              style={{ color: '#C9A84C', fontFamily: 'var(--font-mono)' }}
            >
              ← Back to login
            </Link>
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <Head><title>Request Access — Prime Picks</title></Head>
      <div className="field-bg min-h-screen flex items-center justify-center px-4 py-10">
        <div className="w-full max-w-md">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="text-4xl mb-2">🏈</div>
            <h1 className="score-display text-chalk" style={{ fontSize: 28, letterSpacing: '0.1em' }}>
              REQUEST ACCESS
            </h1>
            <p className="text-slate text-xs mt-1" style={{ fontFamily: 'var(--font-mono)' }}>
              Prime Picks — NFL & NCAAF score prediction
            </p>
          </div>

          <div className="panel rounded-xl p-7">
            <p className="text-sm text-slate mb-6" style={{ fontFamily: 'var(--font-mono)', lineHeight: 1.6 }}>
              Fill out the form below and you'll be notified by email once your access is approved.
            </p>

            <div className="space-y-4">
              {/* First + Last */}
              <div className="grid grid-cols-2 gap-3">
                {[
                  { key: 'firstName', label: 'First Name', placeholder: 'Jay' },
                  { key: 'lastName', label: 'Last Name', placeholder: 'Smith' },
                ].map(f => (
                  <div key={f.key}>
                    <label className="block text-xs text-slate mb-1 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>
                      {f.label} <span style={{ color: '#C9A84C' }}>*</span>
                    </label>
                    <input
                      type="text"
                      className="w-full rounded px-3 py-2.5 text-sm"
                      placeholder={f.placeholder}
                      value={form[f.key] || ''}
                      onChange={e => set(f.key, e.target.value)}
                    />
                  </div>
                ))}
              </div>

              {/* Email */}
              <div>
                <label className="block text-xs text-slate mb-1 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>
                  Email Address <span style={{ color: '#C9A84C' }}>*</span>
                </label>
                <input
                  type="email"
                  className="w-full rounded px-3 py-2.5 text-sm"
                  placeholder="you@example.com"
                  value={form.email || ''}
                  onChange={e => set('email', e.target.value)}
                  autoComplete="email"
                />
              </div>

              {/* Cell Phone */}
              <div>
                <label className="block text-xs text-slate mb-1 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>
                  Cell Phone <span style={{ color: '#4A5568', fontWeight: 400 }}>(optional)</span>
                </label>
                <input
                  type="tel"
                  className="w-full rounded px-3 py-2.5 text-sm"
                  placeholder="(305) 555-0100"
                  value={form.phone || ''}
                  onChange={e => set('phone', e.target.value)}
                />
              </div>

              {/* How did you hear about us */}
              <div>
                <label className="block text-xs text-slate mb-1 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>
                  How did you hear about Prime Picks? <span style={{ color: '#4A5568', fontWeight: 400 }}>(optional)</span>
                </label>
                <select
                  className="w-full rounded px-3 py-2.5 text-sm"
                  value={form.referral || ''}
                  onChange={e => set('referral', e.target.value)}
                >
                  {REFERRAL_OPTIONS.map(o => (
                    <option key={o} value={o === 'Select one...' ? '' : o}>{o}</option>
                  ))}
                </select>
              </div>

              {/* Password */}
              <div>
                <label className="block text-xs text-slate mb-1 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>
                  Choose a Password <span style={{ color: '#C9A84C' }}>*</span>
                </label>
                <input
                  type="password"
                  className="w-full rounded px-3 py-2.5 text-sm"
                  placeholder="8+ characters"
                  value={form.password || ''}
                  onChange={e => set('password', e.target.value)}
                  autoComplete="new-password"
                />
              </div>
            </div>

            {error && (
              <div className="mt-4 px-3 py-2 rounded text-xs" style={{ background: 'rgba(217,64,64,0.1)', color: '#D94040', border: '1px solid rgba(217,64,64,0.2)', fontFamily: 'var(--font-mono)' }}>
                {error}
              </div>
            )}

            <button
              onClick={handleSubmit}
              disabled={submitting}
              className="w-full mt-5 score-display py-3 rounded transition-all"
              style={{
                fontSize: 18,
                letterSpacing: '0.12em',
                background: submitting ? 'rgba(201,168,76,0.2)' : 'linear-gradient(135deg, #C9A84C, #E8C96A)',
                color: submitting ? '#4A5568' : '#030B14',
                border: 'none',
                cursor: submitting ? 'not-allowed' : 'pointer',
              }}
            >
              {submitting ? 'SUBMITTING...' : 'REQUEST ACCESS'}
            </button>

            <div className="gold-line mt-6" />

            <p className="text-center text-xs text-slate mt-5" style={{ fontFamily: 'var(--font-mono)' }}>
              Already have access?{' '}
              <Link href="/login" style={{ color: '#C9A84C' }}>
                Sign in →
              </Link>
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
