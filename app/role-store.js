// Module-level role store — persists for the lifetime of the JS runtime (one session).
// Replace with AsyncStorage if cross-session persistence is needed.
let _role = null;

export const setRole = (role) => {
  _role = role;
};

export const getRole = () => _role;
