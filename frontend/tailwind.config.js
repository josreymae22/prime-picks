/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx}',
    './src/components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        field: {
          900: '#030B14',
          800: '#071524',
          700: '#0A1F33',
          600: '#0F2C47',
          500: '#163D62',
        },
        turf: '#1A4A2E',
        gold: '#C9A84C',
        'gold-light': '#E8C96A',
        chalk: '#F0EEE6',
        slate: '#8B9BB4',
        red: '#D94040',
        green: '#3DAA6A',
      },
      fontFamily: {
        display: ['var(--font-display)', 'serif'],
        body: ['var(--font-body)', 'sans-serif'],
        mono: ['var(--font-mono)', 'monospace'],
      },
    },
  },
  plugins: [],
};
