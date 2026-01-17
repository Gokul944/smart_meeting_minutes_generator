from rule_nlp import summarize, extract_actions

text = """
We discussed the project timeline.
John will complete the backend by Friday.
The UI needs improvement.
We should test the application thoroughly.
Next meeting is on Monday.
"""

print("SUMMARY:")
print(summarize(text, 3))

print("\nACTIONS:")
for a in extract_actions(text):
    print("-", a)
