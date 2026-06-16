import type { NextApiRequest, NextApiResponse } from 'next';
import { initializeApp, getApps, cert } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';
import { getAuth } from 'firebase-admin/auth';
import { Resend } from 'resend';

// Firebase Admin init
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
const adminAuth = getAuth();
const resend = new Resend(process.env.RESEND_API_KEY);

const ADMIN_SECRET = process.env.ADMIN_SECRET || 'changeme';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).end();

  const { secret, uid, action } = req.body; // action: 'approve' | 'deny'

  if (secret !== ADMIN_SECRET) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  if (!uid || !['approve', 'deny'].includes(action)) {
    return res.status(400).json({ error: 'uid and action required' });
  }

  const status = action === 'approve' ? 'approved' : 'denied';

  try {
    // Update Firestore
    await adminDb.collection('users').doc(uid).update({ status });

    // If approving, also enable the Firebase Auth account
    if (action === 'approve') {
      await adminAuth.updateUser(uid, { disabled: false });
    } else {
      await adminAuth.updateUser(uid, { disabled: true });
    }

    // Get user info for email
    const snap = await adminDb.collection('users').doc(uid).get();
    const userData = snap.data();

    if (userData?.email) {
      if (action === 'approve') {
        await resend.emails.send({
          from: 'Prime Picks <onboarding@resend.dev>',
          to: userData.email,
          subject: '✅ Your Prime Picks access has been approved',
          html: `
            <div style="font-family: Arial, sans-serif; background: #030B14; color: #F0EEE6; padding: 32px; border-radius: 12px; max-width: 480px;">
              <div style="font-size: 11px; letter-spacing: 0.15em; color: #C9A84C; text-transform: uppercase; margin-bottom: 8px;">Prime Picks</div>
              <h2 style="margin: 0 0 16px; font-size: 22px;">You're in, ${userData.firstName}.</h2>
              <p style="color: #8B9BB4; font-size: 14px; line-height: 1.6;">
                Your access to Prime Picks has been approved. Log in with the email and password you registered with.
              </p>
              <a href="${process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'}/login"
                 style="display: inline-block; margin-top: 24px; background: #C9A84C; color: #030B14; text-decoration: none; padding: 12px 24px; border-radius: 6px; font-weight: 700; font-size: 14px;">
                Log In Now →
              </a>
            </div>
          `,
        });
      } else {
        await resend.emails.send({
          from: 'Prime Picks <onboarding@resend.dev>',
          to: userData.email,
          subject: 'Prime Picks — Access request update',
          html: `
            <div style="font-family: Arial, sans-serif; background: #030B14; color: #F0EEE6; padding: 32px; border-radius: 12px; max-width: 480px;">
              <div style="font-size: 11px; letter-spacing: 0.15em; color: #C9A84C; text-transform: uppercase; margin-bottom: 8px;">Prime Picks</div>
              <h2 style="margin: 0 0 16px; font-size: 22px;">Access request update</h2>
              <p style="color: #8B9BB4; font-size: 14px; line-height: 1.6;">
                Hi ${userData.firstName}, unfortunately we're unable to approve your access request at this time.
                If you believe this is an error, please reply to this email.
              </p>
            </div>
          `,
        });
      }
    }

    return res.status(200).json({ ok: true, status });
  } catch (err: any) {
    console.error('Approve/deny error:', err);
    return res.status(500).json({ error: err.message });
  }
}
