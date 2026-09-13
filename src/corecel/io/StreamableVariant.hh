//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/io/StreamableVariant.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>
#include <variant>

#include "corecel/Assert.hh"

#include "StreamUtils.hh"

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

    //! Write to stream
    friend std::ostream& operator<<(std::ostream& os,
                                    StreamableVariant const& svar)
    {
        CELER_ASSUME(!svar.value.valueless_by_exception());
        std::visit(GenericToStream{os}, svar.value);
        return os;
    }

    //! Save as a string
    friend std::string to_string(StreamableVariant const& svar)
    {
        return stream_to_string(svar);
    }
};

//---------------------------------------------------------------------------//
// Deduction guide
template<class T>
StreamableVariant(T&&) -> StreamableVariant<T>;

//---------------------------------------------------------------------------//
}  // namespace celeritas
