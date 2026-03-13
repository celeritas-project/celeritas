//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/StreamableVariant.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>
#include <sstream>
#include <utility>
#include <variant>

#include "corecel/Assert.hh"

#include "StreamToString.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Helper class to print a variant to a stream.
 *
 * Example:
 * \code
   std::cout << StreamableVariant{surface} << std::endl;
   \endcode
 */
template<class T>
struct StreamableVariant
{
    T value;
};

//---------------------------------------------------------------------------//
// Deduction guide
template<class T>
StreamableVariant(T&&) -> StreamableVariant<T>;

//---------------------------------------------------------------------------//
// IMPLEMENTATION
//---------------------------------------------------------------------------//
namespace detail
{
struct GenericToStream
{
    std::ostream& os;

    template<class T>
    void operator()(T&& obj) const
    {
        this->os << std::forward<T>(obj);
    }
};
}  // namespace detail

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Write a variant object's value to a stream.
 */
template<class T>
inline std::ostream&
operator<<(std::ostream& os, StreamableVariant<T> const& svar)
{
    CELER_ASSUME(!svar.value.valueless_by_exception());
    std::visit(detail::GenericToStream{os}, svar.value);
    return os;
}

//---------------------------------------------------------------------------//
/*!
 * Save a variant object's value to a string.
 */
template<class T>
inline std::string to_string(StreamableVariant<T> const& svar)
{
    return stream_to_string(svar);
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
