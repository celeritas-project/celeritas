//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/Repr.hh
//---------------------------------------------------------------------------//
#pragma once

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "corecel/OpaqueId.hh"
#include "corecel/cont/Array.hh"
#include "corecel/cont/Span.hh"
#include "corecel/data/Collection.hh"
#include "corecel/math/Quantity.hh"
#include "celeritas/phys/AtomicNumber.hh"

#include "Join.hh"

#include "detail/ReprImpl.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Return a streamable object that prints out a C++-style.
 *
 * Example:
 * \code
  std::cout << repr(my_vec) << std::endl;
  \endcode

 * The 'name' argument defaults to null (just printing the value); if a string
 * is given a full variable declaration such as `std::string foo{"bar"}` will
 * be printed. If the name is empty, an anonymous value `std::string{"bar"}`
 * will be printed.
 */
template<class T>
detail::Repr<T> repr(T const& obj, char const* name = nullptr)
{
    return {obj, name};
}

//---------------------------------------------------------------------------//
/*!
 * Traits for writing an object for diagnostic or testing output.
 *
 * The "streamable" traits usually write so that the object can be injected
 * into test code.  The default tries to use whatever ostream operator is
 * available.
 * Other overrides are provided for collections, characters, and more?
 */
template<class T>
struct ReprTraits
{
    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_simple_type(os, "UNKNOWN", name);
    }
    static void init(std::ostream&) {}
    static void print_value(std::ostream& os, T const& value) { os << value; }
};

template<>
struct ReprTraits<float>
{
    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_simple_type(os, "float", name);
    }
    static void init(std::ostream& os) { os.precision(7); }
    static void print_value(std::ostream& os, float value)
    {
        os << value << 'f';
    }
};

template<>
struct ReprTraits<double>
{
    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_simple_type(os, "double", name);
    }
    static void init(std::ostream& os) { os.precision(14); }
    static void print_value(std::ostream& os, double value) { os << value; }
};

template<>
struct ReprTraits<char>
{
    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_simple_type(os, "char", name);
    }
    static void init(std::ostream&) {}
    static void print_value(std::ostream& os, char value)
    {
        os << '\'';
        detail::repr_char(os, value);
        os << '\'';
    }
};

template<>
struct ReprTraits<unsigned char>
{
    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_simple_type(os, "unsigned char", name);
    }
    static void init(std::ostream& os) { os << std::setfill('0') << std::hex; }
    static void print_value(std::ostream& os, unsigned char value)
    {
        os << "'\\x" << detail::char_to_hex_string(value) << '\'';
    }
};

template<>
struct ReprTraits<bool>
{
    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_simple_type(os, "bool", name);
    }
    static void init(std::ostream& os) { os << std::boolalpha; }
    static void print_value(std::ostream& os, bool value) { os << value; }
};

template<>
struct ReprTraits<int>
{
    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_simple_type(os, "int", name);
    }
    static void init(std::ostream&) {}
    static void print_value(std::ostream& os, int value) { os << value; }
};

template<>
struct ReprTraits<unsigned int>
{
    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_simple_type(os, "unsigned int", name);
    }
    static void init(std::ostream&) {}
    static void print_value(std::ostream& os, unsigned int value)
    {
        os << value << 'u';
    }
};

template<>
struct ReprTraits<long>
{
    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_simple_type(os, "long", name);
    }
    static void init(std::ostream&) {}
    static void print_value(std::ostream& os, long value)
    {
        os << value << 'l';
    }
};

template<>
struct ReprTraits<unsigned long>
{
    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_simple_type(os, "unsigned long", name);
    }
    static void init(std::ostream&) {}
    static void print_value(std::ostream& os, unsigned long value)
    {
        os << value << "ul";
    }
};

template<>
struct ReprTraits<long long>
{
    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_simple_type(os, "long long", name);
    }
    static void init(std::ostream&) {}
    static void print_value(std::ostream& os, long long value)
    {
        os << value << "ll";
    }
};

template<>
struct ReprTraits<unsigned long long>
{
    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_simple_type(os, "unsigned long long", name);
    }
    static void init(std::ostream&) {}
    static void print_value(std::ostream& os, unsigned long long value)
    {
        os << value << "ull";
    }
};

template<>
struct ReprTraits<std::string_view>
{
    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_simple_type(os, "std::string_view", name);
    }

    static void init(std::ostream&) {}
    static void print_value(std::ostream& os, std::string_view value)
    {
        using ssize = std::streamsize;
        ssize width = os.width();
        os.width(
            std::max(width - static_cast<ssize>(value.size()) - 2, ssize{0}));

        if (detail::all_printable(value)
            && (value.size() > 70
                || std::count(value.begin(), value.end(), '"') > 2))
        {
            // Print long literal strings or ones with many quotes on one line
            os << "R\"(" << value << ")\"";
            return;
        }

        os << '"';
        for (char c : value)
        {
            if (c == '\"')
            {
                os << '\\';
            }
            detail::repr_char(os, c);
        }
        os << '"';
    }
};

template<>
struct ReprTraits<std::string>
{
    using value_type = std::string::value_type;

    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_simple_type(os, "std::string", name);
    }

    static void init(std::ostream&) {}
    static void print_value(std::ostream& os, std::string const& value)
    {
        ReprTraits<std::string_view>::print_value(os, value);
    }
};

template<>
struct ReprTraits<char*>
{
    using value_type = char;

    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_simple_type(os, "char const*", name);
    }

    static void init(std::ostream&) {}

    static void print_value(std::ostream& os, char const* value)
    {
        if (value)
        {
            ReprTraits<std::string_view>::print_value(os, value);
        }
        else
        {
            os << "nullptr";
        }
    }
};

template<std::size_t N>
struct ReprTraits<char[N]> : ReprTraits<char*>
{
};

template<>
struct ReprTraits<char const*> : ReprTraits<char*>
{
};

//! Specialization for printing std::pairs
template<class T1, class T2>
struct ReprTraits<std::pair<T1, T2>>
{
    using RT1 = ReprTraits<T1>;
    using RT2 = ReprTraits<T2>;

    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        os << "std::pair<";
        RT1::print_type(os);
        os << ',';
        RT2::print_type(os);
        os << '>';
        if (name)
        {
            os << ' ' << name;
        }
    }

    static void init(std::ostream& os)
    {
        RT1::init(os);
        RT2::init(os);
    }

    static void print_value(std::ostream& os, std::pair<T1, T2> const& value)
    {
        os << '{';
        RT1::print_value(os, value.first);
        os << ',';
        RT2::print_value(os, value.second);
        os << '}';
    }
};

//! Specialization for AtomicNumber
template<>
struct ReprTraits<AtomicNumber>
{
    using RT = ReprTraits<int>;

    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        os << "AtomicNumber";
        if (name)
        {
            os << ' ' << name;
        }
    }

    static void init(std::ostream& os) { RT::init(os); }

    static void print_value(std::ostream& os, AtomicNumber const& value)
    {
        os << '{';
        if (value)
        {
            RT::print_value(os, value.unchecked_get());
        }
        os << '}';
    }
};

//! Specialization for OpaqueId
template<class V, class S>
struct ReprTraits<OpaqueId<V, S>>
{
    using RT = ReprTraits<S>;

    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        os << "OpaqueID<?>";
        if (name)
        {
            os << ' ' << name;
        }
    }

    static void init(std::ostream& os) { RT::init(os); }

    static void print_value(std::ostream& os, OpaqueId<V, S> const& value)
    {
        os << '{';
        if (value)
        {
            RT::print_value(os, value.unchecked_get());
        }
        os << '}';
    }
};

//! Specialization for Quantity
template<class U, class V>
struct ReprTraits<Quantity<U, V>>
{
    using RT = ReprTraits<V>;

    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        os << "Quantity<?,?>";
        if (name)
        {
            os << ' ' << name;
        }
    }

    static void init(std::ostream& os) { RT::init(os); }

    static void print_value(std::ostream& os, Quantity<U, V> const& q)
    {
        os << '{';
        RT::print_value(os, q.value());
        os << '}';
    }
};

//---------------------------------------------------------------------------//
// CONTAINER TRAITS
//---------------------------------------------------------------------------//
template<class Container>
struct ContTraits
{
    using size_type = typename Container::size_type;
    using value_type = std::decay_t<typename Container::value_type>;
};

template<class T, std::size_t N>
struct ContTraits<T[N]>
{
    using size_type = std::size_t;
    using value_type = std::decay_t<T>;
};

/*!
 * Get a string representation of a container of type T.
 */
template<class Container>
struct ContainerReprTraits
{
    using value_type = typename ContTraits<Container>::value_type;
    using RT = ReprTraits<value_type>;

    static void init(std::ostream& os) { RT::init(os); }

    static void print_value(std::ostream& os, Container const& data)
    {
        os << '{'
           << join_stream(
                  std::begin(data), std::end(data), ", ", RT::print_value);
        if (std::size(data) > 8)
        {
            // Add a trailing ',' to long outputs to help clang-format
            os << ',';
        }
        os << '}';
    }
};

template<class T, class A>
struct ReprTraits<std::vector<T, A>>
    : public ContainerReprTraits<std::vector<T, A>>
{
    using value_type = std::decay_t<T>;

    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_container_type<value_type>(os, "std::vector", name);
    }
};

template<class T, size_type N>
struct ReprTraits<Array<T, N>> : public ContainerReprTraits<Array<T, N>>
{
    using value_type = std::decay_t<T>;

    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        os << "Array<";
        ReprTraits<value_type>::print_type(os);
        os << ',' << N << '>';
        if (name)
        {
            os << ' ' << name;
        }
    }
};

template<class T, std::size_t N>
struct ReprTraits<T[N]> : public ContainerReprTraits<T[N]>
{
    using value_type = std::decay_t<T>;

    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        os << "Array<";
        ReprTraits<value_type>::print_type(os);
        os << ',' << N << '>';
        if (name)
        {
            os << ' ' << name;
        }
    }
};

template<class T, std::size_t N>
struct ReprTraits<Span<T, N>> : public ContainerReprTraits<Span<T, N>>
{
    using value_type = std::decay_t<T>;

    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_container_type<value_type>(os, "Span", name);
    }
};

//! Print collection host data
template<class T, Ownership W, class I>
struct ReprTraits<Collection<T, W, MemSpace::host, I>>
{
    using ContainerT = Collection<T, W, MemSpace::host, I>;
    using value_type = typename ContainerT::value_type;
    using RT = ReprTraits<value_type>;

    static void init(std::ostream& os) { RT::init(os); }

    static void print_value(std::ostream& os, ContainerT const& data)
    {
        auto view = data[typename ContainerT::AllItemsT{}];
        os << '{'
           << join_stream(
                  std::begin(view), std::end(view), ", ", RT::print_value)
           << '}';
    }

    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_container_type<value_type>(os, "Collection", name);
    }
};

//! Print placeholder for device data
template<class T, Ownership W, class I>
struct ReprTraits<Collection<T, W, MemSpace::device, I>>
{
    using ContainerT = Collection<T, W, MemSpace::device, I>;
    using value_type = typename ContainerT::value_type;

    static void init(std::ostream&) {}

    static void print_value(std::ostream& os, ContainerT const& data)
    {
        auto view = data[typename ContainerT::AllItemsT{}];
        os << "<device collection, size=" << data.size() << '>';
    }

    static void print_type(std::ostream& os, char const* name = nullptr)
    {
        detail::print_container_type<value_type>(os, "Collection", name);
    }
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
