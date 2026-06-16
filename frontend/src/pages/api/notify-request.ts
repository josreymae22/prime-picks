import type { NextApiRequest, NextApiResponse } from 'next';
import { Resend } from 'resend';

const resend = new Resend(process.env.RESEND_API_KEY);
const ADMIN_EMAIL = 'jay.rathman.sports@gmail.com';
const APP_URL = process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== 'POST') return res.status(405).end();

  const { firstName, lastName, email, phone, referral, uid } = req.body;

  if (!firstName || !lastName || !email || !uid) {
    return res.status(400).json({ error: 'Missing required fields' });
  }

  try {
    await resend.emails.send({
      from: 'Prime Picks <onboarding@resend.dev>',
      to: ADMIN_EMAIL,
      subject: `🏈 New Access Request — ${firstName} ${lastName}`,
      html: `
        <div style="font-family: 'IBM Plex Sans', Arial, sans-serif; background: #030B14; color: #F0EEE6; padding: 32px; border-radius: 12px; max-width: 520px;">
          <div style="font-size: 11px; letter-spacing: 0.15em; color: #C9A84C; text-transform: uppercase; margin-bottom: 8px;">Prime Picks</div>
          <h2 style="margin: 0 0 24px; font-size: 22px; color: #F0EEE6;">New Access Request</h2>

          <table style="width: 100%; border-collapse: collapse;">
            <tr><td style="padding: 8px 0; color: #8B9BB4; font-size: 13px; width: 140px;">Name</td><td style="padding: 8px 0; color: #F0EEE6; font-size: 13px;">${firstName} ${lastName}</td></tr>
            <tr><td style="padding: 8px 0; color: #8B9BB4; font-size: 13px;">Email</td><td style="padding: 8px 0; color: #F0EEE6; font-size: 13px;">${email}</td></tr>
            <tr><td style="padding: 8px 0; color: #8B9BB4; font-size: 13px;">Phone</td><td style="padding: 8px 0; color: #F0EEE6; font-size: 13px;">${phone || '—'}</td></tr>
            <tr><td style="padding: 8px 0; color: #8B9BB4; font-size: 13px;">Found us via</td><td style="padding: 8px 0; color: #F0EEE6; font-size: 13px;">${referral || '—'}</td></tr>
          </table>

          <div style="margin-top: 28px;">
            <a href="${APP_URL}/admin?action=approve&uid=${uid}"
               style="display: inline-block; margin-right: 12px; background: #3DAA6A; color: #fff; text-decoration: none; padding: 10px 20px; border-radius: 6px; font-size: 13px; font-weight: 600;">
              ✓ Approve Access
            </a>
            <a href="${APP_URL}/admin?action=deny&uid=${uid}"
               style="display: inline-block; background: #D94040; color: #fff; text-decoration: none; padding: 10px 20px; border-radius: 6px; font-size: 13px; font-weight: 600;">
              ✗ Deny Access
            </a>
          </div>

          <p style="margin-top: 20px; font-size: 11px; color: #4A5568;">
            Or manage all users at <a href="${APP_URL}/admin" style="color: #C9A84C;">${APP_URL}/admin</a>
          </p>
        </div>
      `,
    });

    return res.status(200).json({ ok: true });
  } catch (err: any) {
    console.error('Email send error:', err);
    return res.status(500).json({ error: 'Failed to send notification' });
  }
}
