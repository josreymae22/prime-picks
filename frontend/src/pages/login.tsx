import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Link from 'next/link';
import Head from 'next/head';
import { useAuth } from '../context/AuthContext';

export default function Login() {
  const { signIn, user, userStatus, loading } = useAuth();
  const router = useRouter();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!loading && user && userStatus === 'approved') {
      router.replace('/');
    }
  }, [user, userStatus, loading, router]);

  const handleLogin = async () => {
    if (!email || !password) { setError('Enter your email and password.'); return; }
    setError('');
    setSubmitting(true);
    try {
      await signIn(email, password);
      // redirect handled by useEffect
    } catch (e: any) {
      const code = e?.code;
      if (code === 'auth/invalid-credential' || code === 'auth/wrong-password' || code === 'auth/user-not-found') {
        setError('Invalid email or password.');
      } else if (code === 'auth/user-disabled') {
        setError('Your account is pending approval.');
      } else {
        setError('Login failed. Try again.');
      }
    } finally {
      setSubmitting(false);
    }
  };

  // Show pending/denied state post-login
  if (!loading && user && userStatus === 'pending') {
    return (
      <PendingScreen email={user.email || ''} onSignOut={async () => {
        const { signOut } = useAuth();
        await signOut();
      }} />
    );
  }

  return (
    <>
      <Head><title>Login — Prime Picks</title></Head>
      <div className="field-bg min-h-screen flex items-center justify-center px-4">
        <div className="w-full max-w-sm">
          {/* Logo */}
          <div className="text-center mb-10">
            <div className="text-4xl mb-2">🏈</div>
            <h1 className="score-display text-chalk" style={{ fontSize: 32, letterSpacing: '0.1em' }}>
              PRIME PICKS
            </h1>
            <p className="text-slate text-xs mt-1" style={{ fontFamily: 'var(--font-mono)' }}>
              Score prediction engine
            </p>
          </div>

          <div className="panel rounded-xl p-7">
            <h2 className="text-chalk text-sm font-semibold mb-5 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>
              Sign In
            </h2>

            <div className="space-y-4">
              <div>
                <label className="block text-xs text-slate mb-1 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>
                  Email
                </label>
                <input
                  type="email"
                  className="w-full rounded px-3 py-2.5 text-sm"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleLogin()}
                  placeholder="you@example.com"
                  autoComplete="email"
                />
              </div>
              <div>
                <label className="block text-xs text-slate mb-1 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>
                  Password
                </label>
                <input
                  type="password"
                  className="w-full rounded px-3 py-2.5 text-sm"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleLogin()}
                  placeholder="••••••••"
                  autoComplete="current-password"
                />
              </div>
            </div>

            {error && (
              <div className="mt-3 px-3 py-2 rounded text-xs" style={{ background: 'rgba(217,64,64,0.1)', color: '#D94040', border: '1px solid rgba(217,64,64,0.2)' }}>
                {error}
              </div>
            )}

            <button
              onClick={handleLogin}
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
              {submitting ? 'SIGNING IN...' : 'SIGN IN'}
            </button>

            <div className="gold-line mt-6" />

            <p className="text-center text-xs text-slate mt-5" style={{ fontFamily: 'var(--font-mono)' }}>
              Don't have access?{' '}
              <Link href="/request-access" className="text-gold" style={{ color: '#C9A84C' }}>
                Request Access →
              </Link>
            </p>
          </div>
        </div>
      </div>
    </>
  );
}

function PendingScreen({ email }: { email: string }) {
  const { signOut } = useAuth();
  return (
    <>
      <Head><title>Pending Approval — Prime Picks</title></Head>
      <div className="field-bg min-h-screen flex items-center justify-center px-4">
        <div className="panel rounded-xl p-8 max-w-sm w-full text-center">
          <div className="text-4xl mb-4">⏳</div>
          <h2 className="score-display text-chalk mb-2" style={{ fontSize: 24, letterSpacing: '0.08em' }}>
            ACCESS PENDING
          </h2>
          <p className="text-slate text-sm mb-2" style={{ fontFamily: 'var(--font-mono)' }}>
            Your request is under review.
          </p>
          <p className="text-slate text-xs mb-6" style={{ fontFamily: 'var(--font-mono)' }}>
            We'll email <span style={{ color: '#C9A84C' }}>{email}</span> once approved.
          </p>
          <button
            onClick={() => signOut()}
            className="text-xs text-slate underline"
            style={{ background: 'none', border: 'none', cursor: 'pointer' }}
          >
            Sign out
          </button>
        </div>
      </div>
    </>
  );
}
