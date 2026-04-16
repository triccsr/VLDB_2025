#pragma once
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <stdexcept>

template <typename T>
class ArrayOnHeap {
   private:
    size_t m_size;
    T* m_data;

   public:
    using iterator = T*;
    using const_iterator = const T*;

    struct uninitialized_t {};
    static constexpr uninitialized_t uninitialized{};

    ArrayOnHeap() : m_size(0), m_data(nullptr) {}
    explicit ArrayOnHeap(size_t n) : m_size(n), m_data(new T[n]()) {}
    ArrayOnHeap(size_t n, uninitialized_t) : m_size(n), m_data(new T[n]) {}
    ArrayOnHeap(size_t n, const T& value) : m_size(n), m_data(new T[n]) { std::fill(m_data, m_data + n, value); }

    ArrayOnHeap(std::initializer_list<T> list) : m_size(list.size()), m_data(new T[list.size()]) {
        std::copy(list.begin(), list.end(), m_data);
    }

    template <std::input_iterator InputIt>
    ArrayOnHeap(InputIt first, InputIt last) : m_size(std::distance(first, last)), m_data(new T[m_size]) {
        std::copy(first, last, m_data);
    }

    ~ArrayOnHeap() { delete[] m_data; }

    ArrayOnHeap(const ArrayOnHeap& other) : m_size(other.m_size), m_data(new T[other.m_size]) {
        std::copy(other.m_data, other.m_data + m_size, m_data);
    }

    ArrayOnHeap& operator=(const ArrayOnHeap& other) {
        ArrayOnHeap tmp(other);
        swap(tmp);
        return *this;
    }

    ArrayOnHeap(ArrayOnHeap&& other) noexcept : m_size(other.m_size), m_data(other.m_data) {
        other.m_data = nullptr;
        other.m_size = 0;
    }

    ArrayOnHeap& operator=(ArrayOnHeap&& other) noexcept {
        if (this != &other) {
            delete[] m_data;
            m_size = other.m_size;
            m_data = other.m_data;
            other.m_data = nullptr;
            other.m_size = 0;
        }
        return *this;
    }

    bool operator==(const ArrayOnHeap& other) const {
        if (m_size != other.m_size) return false;
        for (size_t i = 0; i < m_size; ++i) {
            if (m_data[i] != other.m_data[i]) return false;
        }
        return true;
    }
    bool operator!=(const ArrayOnHeap& other) const { return !(*this == other); }

    void assign(size_t n, const T& value) {
        T* newData = new T[n];
        std::fill(newData, newData + n, value);
        delete[] m_data;
        m_data = newData;
        m_size = n;
    }

    void swap(ArrayOnHeap& other) noexcept {
        std::swap(m_size, other.m_size);
        std::swap(m_data, other.m_data);
    }

    T& operator[](size_t index) { return m_data[index]; }
    const T& operator[](size_t index) const { return m_data[index]; }

    T& at(size_t index) {
        if (index >= m_size) throw std::out_of_range("Index out of range");
        return m_data[index];
    }
    const T& at(size_t index) const {
        if (index >= m_size) throw std::out_of_range("Index out of range");
        return m_data[index];
    }

    size_t size() const { return m_size; }
    bool empty() const { return m_size == 0; }

    T& front() { return m_data[0]; }
    const T& front() const { return m_data[0]; }
    T& back() { return m_data[m_size - 1]; }
    const T& back() const { return m_data[m_size - 1]; }

    T* data() { return m_data; }
    const T* data() const { return m_data; }

    iterator begin() { return m_data; }
    iterator end() { return m_data + m_size; }
    const_iterator begin() const { return m_data; }
    const_iterator end() const { return m_data + m_size; }
    const_iterator cbegin() const { return m_data; }
    const_iterator cend() const { return m_data + m_size; }

    void memset(int value) { std::memset(m_data, value, m_size * sizeof(T)); }
};
