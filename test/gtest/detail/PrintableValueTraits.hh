//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file PrintableValueTraits.hh
//---------------------------------------------------------------------------//
#ifndef test_gtest_detail_PrintableValueTraits_hh
#define test_gtest_detail_PrintableValueTraits_hh

#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include "Utils.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
template<class Container>
inline std::string to_string(const Container& data);

//---------------------------------------------------------------------------//
struct StreamableChar
{
    char c;
};

inline std::ostream& operator<<(std::ostream& os, StreamableChar sc)
{
    if (std::isprint(sc.c))
    {
        os << sc.c;
    }
    else
    {
        os << '\\';
        switch (sc.c)
        {
            case 0:
                os << '0';
                break;
            case '\n':
                os << 'n';
                break;
            case '\t':
                os << 't';
                break;
            case '\r':
                os << 'r';
                break;
            default:
                os << 'x'
                   << char_to_hex_string(static_cast<unsigned char>(sc.c));
                break;
        }
    }
    return os;
}

//---------------------------------------------------------------------------//
template<class Container>
struct ContTraits
{
    using size_type = typename Container::size_type;
    using value_type =
        typename std::decay<typename Container::value_type>::type;
};

template<class T, std::size_t N>
struct ContTraits<T[N]>
{
    using size_type  = std::size_t;
    using value_type = typename std::decay<T>::type;
};

//! Helper class for printing values.
template<class T>
struct PrintableValueTraits
{
    static const char* name() { return "UNKNOWN"; }
    static void        init(std::ostream&) {}
    static void        print(std::ostream& os, const T& value) { os << value; }
};

template<>
struct PrintableValueTraits<float>
{
    static const char* name() { return "float"; }
    static void        init(std::ostream& os) { os.precision(7); }
    static void print(std::ostream& os, float value) { os << value << 'f'; }
};

template<>
struct PrintableValueTraits<double>
{
    static const char* name() { return "double"; }
    static void        init(std::ostream& os) { os.precision(13); }
    static void        print(std::ostream& os, double value) { os << value; }
};

template<>
struct PrintableValueTraits<char>
{
    static const char* name() { return "char"; }
    static void        init(std::ostream&) {}
    static void        print(std::ostream& os, char value)
    {
        os << '\'' << StreamableChar{value} << '\'';
    }
};

template<>
struct PrintableValueTraits<unsigned char>
{
    static const char* name() { return "unsigned char"; }
    static void init(std::ostream& os) { os << std::setfill('0') << std::hex; }
    static void print(std::ostream& os, unsigned char value)
    {
        os << "0x" << char_to_hex_string(value);
    }
};

template<>
struct PrintableValueTraits<bool>
{
    static const char* name() { return "bool"; }
    static void        init(std::ostream& os) { os << std::boolalpha; }
    static void        print(std::ostream& os, bool value) { os << value; }
};

template<>
struct PrintableValueTraits<int>
{
    static const char* name() { return "int"; }
    static void        init(std::ostream&) {}
    static void        print(std::ostream& os, int value) { os << value; }
};

template<>
struct PrintableValueTraits<unsigned int>
{
    static const char* name() { return "unsigned int"; }
    static void        init(std::ostream&) {}
    static void        print(std::ostream& os, unsigned int value)
    {
        os << value << 'u';
    }
};

#ifndef _WIN32
template<>
struct PrintableValueTraits<long>
{
    static const char* name() { return "long"; }
    static void        init(std::ostream&) {}
    static void print(std::ostream& os, long value) { os << value << 'l'; }
};

template<>
struct PrintableValueTraits<unsigned long>
{
    static const char* name() { return "unsigned long"; }
    static void        init(std::ostream&) {}
    static void        print(std::ostream& os, unsigned long value)
    {
        os << value << "ul";
    }
};
#endif

template<>
struct PrintableValueTraits<long long>
{
    static const char* name() { return "long long"; }
    static void        init(std::ostream&) {}
    static void        print(std::ostream& os, long long value)
    {
        os << value << "ll";
    }
};

template<>
struct PrintableValueTraits<unsigned long long>
{
    static const char* name() { return "unsigned long long"; }
    static void        init(std::ostream&) {}
    static void        print(std::ostream& os, unsigned long long value)
    {
        os << value << "ull";
    }
};

template<>
struct PrintableValueTraits<std::string>
{
    static const char* name() { return "std::string"; }

    static void init(std::ostream&) {}
    static void print(std::ostream& os, const std::string& value)
    {
        std::streamsize width = os.width();

        os.width(width - value.size() - 2);

        os << '"';
        for (char c : value)
        {
            os << StreamableChar{c};
        }
        os << '"';
    }
};

template<>
struct PrintableValueTraits<const char*>
{
    static const char* name() { return "const char*"; }

    static void init(std::ostream&) {}

    static void print(std::ostream& os, const char* value)
    {
        if (value)
        {
            PrintableValueTraits<std::string>::print(os, value);
        }
        else
        {
            os << "nullptr";
        }
    }
};

//! Specialization for printing std::pairs
template<class T1, class T2>
struct PrintableValueTraits<std::pair<T1, T2>>
{
    using PVT1 = PrintableValueTraits<T1>;
    using PVT2 = PrintableValueTraits<T2>;

    static std::string name()
    {
        std::ostringstream os;
        os << "std::pair<" << PVT1::name() << ',' << PVT2::name() << '>';
        return os.str();
    }

    static void init(std::ostream& os)
    {
        PVT1::init(os);
        PVT2::init(os);
    }

    static void print(std::ostream& os, const std::pair<T1, T2>& value)
    {
        os << '{';
        PVT1::print(os, value.first);
        os << ',';
        PVT2::print(os, value.second);
        os << '}';
    }
};

//---------------------------------------------------------------------------//
// Get a string representation of a container
template<class Container>
std::string to_string(const Container& data)
{
    using value_type = typename ContTraits<Container>::value_type;
    using PVT        = PrintableValueTraits<value_type>;

    std::ostringstream os;
    PVT::init(os);

    os << '{';

    auto iter     = std::begin(data);
    auto end_iter = std::end(data);
    if (iter != end_iter)
    {
        PVT::print(os, *iter++);
    }
    while (iter != end_iter)
    {
        os << ", ";
        PVT::print(os, *iter++);
    }
    os << '}';
    return os.str();
}

//---------------------------------------------------------------------------//
} // namespace detail
} // namespace celeritas

#endif // test_gtest_detail_PrintableValueTraits_hh
