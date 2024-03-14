//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/StreamableVariant.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>
#include <variant>

#include "corecel/Assert.hh"

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
        this->os << obj;
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
std::ostream& operator<<(std::ostream& os, StreamableVariant<T> const& svar)
{
    CELER_ASSUME(!svar.value.valueless_by_exception());
    std::visit(detail::GenericToStream{os}, svar.value);
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
