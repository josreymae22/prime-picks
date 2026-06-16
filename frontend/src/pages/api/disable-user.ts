import type { NextApiRequest, NextApiResponse } from 'next';
import { initializeApp, getApps, cert } from 'firebase-admin/app';
import { getAuth } from 'firebase-admin/auth';

if (!getApps().length) {
  initializeApp({
    credential: cert({
      projectId: process.env.FIREBASE_PROJECT_ID,
      clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
      privateKey: process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, '\n'),
    }),
  });
}

const adminAuth = getAuth();

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).end();

  const { uid } = req.body;
  if (!uid) return res.status(400).json({ error: 'uid required' });

  try {
    await adminAuth.updateUser(uid, { disabled: true });
    return res.status(200).json({ ok: true });
  } catch (err: any) {
    console.error('disable-user error:', err);
    return res.status(500).json({ error: err.message });
  }
}
