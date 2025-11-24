# Negative Sampling Strategy

For each manifest row:

Generate:

1× N (gold, your existing spec)

1× S (bland, generic, rule-breaking brochure)

1× E (logistics-heavy, story-light)

1× W (story-heavy, logistics-light)

Optional 2–4× trait-ablations by editing N (no sensory, no indigenous, etc.)

Run them through the big model, capture activations.


Compute:

N_dir = mean(N − S)

EW_dir = mean(E − W)

Trait_dirs = mean(N − N_without_trait) for each trait


Use those:

To steer the small model (adding/subtracting scaled directions)

To filter / project out directions associated with non-essential traits.

## More Insights

### Be very intentional about what you align

Instead of “align the whole hidden state”, align the subspace that’s actually about this task.

a. Build task-specific subspaces first

From your N / S / E / W and trait ablations:

For each pair (same prompt, different style):


Student: same but with student activations.

PCA / SVD to get the main tour-related subspace in the teacher.

Only align these low-dimensional subspaces (e.g. top 32–128 components), not the entire residual stream.

This helps you avoid:

Rotating huge chunks of space that have nothing to do with tours.

Overfitting to noise in a tiny dataset.

This matches how people use Procrustes to align embedding spaces across models / time periods.



Use weighted Procrustes / SVD

Procrustes is usually just minimize over orthogonal 𝑅 but you’ve got richer structure:

Some examples are positive (N, trait-preserving)

Some are negative (S, ablations)


You can:

Form a single stacked matrix of paired directions

Rows = [N−S, E−W, trait diffs, etc.]

Assign weights:

High weight to:

N−S (core “good vs bad tour”)

Lower weight to:

E−W (if you want to preserve both “logistics” and “story” as a controllable axis rather than fully aligning)

Solve a weighted Procrustes problem or, simpler, rescale rows of A and B before computing SVD.

This bakes your “what matters most” into the rotation itself instead of treating all pairs equally.



## FAQ

Why not generate JSON, or assembly? And use that as a negative example?

“Near” vs “far” negatives
What we designed above are near-manifold negatives:
Same task (“be a tour guide for this location”)
Same format (short narrative blurb)
Only one or two capabilities flipped (no anecdotes, no sensory, etc.)
Those are gold for:
Getting clean N−S directions that correspond to tour-guide-ness, not “completely different activity”.
Letting you say: “this neuron activity is about sensory detail, that one about logistics, etc.”

Now compare that with:
JSON response
Assembly code
Some tool-spec format, etc.

Those are far-manifold negatives:
Different discourse structure
Different token distribution
Often a different task entirely (specification / code-gen instead of narrative guidance)
Those will give you very strong directions like:
“Narrative prose vs structured machine-readable output”
“Natural language vs code”


…but they won’t tell you much about:
Good vs bad tour guidance
Rich vs bland sensory description
Authentic vs tourist-trappy recs
So for your tour-guide steering, JSON/asm won’t replace the structured N/S/E/W we talked about.



Is there anything like this out there already?

I haven’t seen an activation-steering paper whose primary contribution is:

“Use ActAdd-style directions to compute an orthogonal transform, then reparameterize the Student’s weights once so there is zero inference overhead.”

Closest neighbors I see are about:

change-of-basis schemes for efficiency inside a single model (not for alignment / distillation),

or standard feature distillation with learned losses, not closed-form Procrustes.


You’re doing something more structured:

For a specific task (tour guide), you design:

N/S (great tour vs deliberately-bad brochure)

E/W (logistics-heavy vs story-heavy)

plus trait-specific ablations (sensory vs no-sensory, indigenous vs not, etc.).

You explicitly treat these as orthogonal-ish axes to define a task subspace in the Teacher.

Then you align the Student’s corresponding subspace to that Teacher subspace via Procrustes, using those carefully tuned contrasts.

That’s a tighter, more “geometric distillation” view than ActAdd, which is mostly “we found a vector; adding it helps”.