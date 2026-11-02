# TriPay Payment Gateway Settlement & Clearing Rules

## Clearing Cycles
TriPay processes QRIS and Virtual Account payments with standard banking clearing intervals based on Bank Indonesia operating hours.

## Settlement Tranches
- **Tranche A (Weekday Transactions):**
  - Transactions occurring Monday through Thursday settle on a T+1 business day cycle.
  - Funds clear into the merchant balance around 14:00 WIB the following business day and become eligible for instant bank disbursement.
- **Tranche B (Weekend & Holiday Transactions):**
  - Transactions occurring Friday afternoon, Saturday, or Sunday cannot clear over the weekend due to central bank clearinghouse closures.
  - Clearing processing starts on Monday morning, and settlement unlocks on Tuesday afternoon after 14:00 WIB.

## Minimum Withdrawal Thresholds
- Minimum balance payout is Rp 50,000 per transaction.
- Standard bank transfer fees (BI-FAST) of Rp 2,500 apply to each outgoing withdrawal.
