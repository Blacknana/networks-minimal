template <class F, class U, class T> auto op_reduce(F func, U arg, T op) {
  return func(arg, op);
}

template <class F, class U, class T, class... Ts>
auto op_reduce(F func, U arg, T first, Ts... rest) {
  return op_reduce(func, func(arg, first), rest...);
}

template <class F, class T> void op_map(F func, T &op) { func(op); }

template <class F, class T, class... Ts>
void op_map(F func, T &first, Ts &...rest) {
  func(first);
  op_map(func, rest...);
}