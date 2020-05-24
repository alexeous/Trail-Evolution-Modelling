#pragma once

inline int divceil(int a, int b) {
	return (int)ceilf((float)a / (float)b);
}

template<typename T> 
inline int sign(T val) { return (T(0) < val) - (val < T(0)); }