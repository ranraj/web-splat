
The floor tilt angle from horizontal is simply:

$$\theta = \arccos(\vec{up} \cdot \hat{Y}) \times \frac{180}{\pi}$$

Where $\hat{Y} = (0, 1, 0)$ is world-up. Since `up.dot(Y) == up.y`, the formula reduces to `acos(up.y)` in degrees.

**How to expose this**: I can add a `#[wasm_bindgen]` method to `WindowContext` that returns the floor angle in degrees, callable from the browser console. Would that be useful, or would you prefer to use it to **automatically derive `init_pitch`** from the actual PLY data instead of the hardcoded `0.096`?