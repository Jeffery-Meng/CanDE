#include <iostream>

template<typename... Targs>
unsigned char compact_bool_recur(unsigned char compact, bool cur, Targs... Fargs) {
  compact <<= 1;
  compact |= static_cast<char>(cur);
  if constexpr (sizeof...(Targs) > 0) {
    compact = compact_bool_recur(compact, Fargs...);
  }
  return compact;
}

// Compacts its arguments into a single byte.
template<typename... Targs>
unsigned char compact_bool(Targs... Fargs) {
  unsigned char compact = 0;
  constexpr size_t sz = sizeof...(Targs);
  static_assert(sz <= sizeof(unsigned char) * 8);
  compact = compact_bool_recur(compact, Fargs...);
  compact <<= (sizeof(unsigned char) * 8 - sz);
  return compact;
}

using namespace std;

int main() {
  cout << static_cast<int>(compact_bool(true)) << endl;
  cout << static_cast<int>(compact_bool(false, true)) << endl;
  cout << static_cast<int>(compact_bool(true, false, true)) << endl;
  cout << static_cast<int>(compact_bool(true, false, true, false)) << endl;
  cout << static_cast<int>(compact_bool(false, false, true, false, true)) << endl;
  cout << static_cast<int>(compact_bool(true, false, true, true, false, true)) << endl;
  cout << static_cast<int>(compact_bool(true, false, true, false, true, false, true)) << endl;
  cout << static_cast<int>(compact_bool(true, true, false, true, true, false, true)) << endl;
}
