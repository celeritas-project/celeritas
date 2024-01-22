//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/QuantityIO.hh
//---------------------------------------------------------------------------//
#pragma once

#include <ostream>

#include "Quantity.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Output an quantity with its label.
 */
template<class UnitT, class ValueT>
std::ostream& operator<<(std::ostream& os, Quantity<UnitT, ValueT> const& q)
{
    static_assert(sizeof(UnitT::label()) > 0,
                  "Unit does not have a 'label' definition");
    os << q.value() << " [" << UnitT::label() << ']';
    return os;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
