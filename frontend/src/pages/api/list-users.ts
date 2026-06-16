import type { NextApiRequest, NextApiResponse } from 'next';
import { initializeApp, getApps, cert } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';

if (!getApps().length) {
  initializeApp({
    credential: cert({
      projectId: process.env.FIREBASE_PROJECT_ID,
      clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
      privateKey: process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, '\n'),
    }),
  });
}

const adminDb = getFirestore();
const ADMIN_SECRET = process.env.ADMIN_SECRET || 'changeme';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).end();

  const { secret } = req.body;
  if (secret !== ADMIN_SECRET) return res.status(401).json({ error: 'Unauthorized' });

  try {
    const snap = await adminDb.collection('users').orderBy('createdAt', 'desc').get();
    const users = snap.docs.map(d => ({ uid: d.id, ...d.data() }));
    return res.status(200).json({ users });
  } catch (err: any) {
    return res.status(500).json({ error: err.message });
  }
}
