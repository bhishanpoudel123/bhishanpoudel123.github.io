# quotations
Today I knew that famous German mathematician Karl Weierstrass said:
"When I wrote this, only God and I understood what I was doing. Now, God only knows."

# using t-sql in stack exchange
```sql
SELECT p.Title, p.Id AS [Post Link], u.Id AS [User Link],p.Score, p.ViewCount, u.DisplayName
FROM Posts p
JOIN Users u ON p.OwnerUserId = u.Id
WHERE p.PostTypeId = 1
AND u.DisplayName = 'BhishanPoudel'
AND p.Score > 10
```