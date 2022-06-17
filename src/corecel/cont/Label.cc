//----------------------------------*-C++-*----------------------------------//
// Copyright 2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/cont/Label.cc
//---------------------------------------------------------------------------//
#include "Label.hh"

#include <ostream>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Write a label to a stream.
 *
 * \todo account for \c os.width .
 */
std::ostream& operator<<(std::ostream& os, const Label& lab)
{
    os << lab.name;

    if (lab.ext.empty())
    {
        // No extension: don't add '@' or anything
        return os;
    }
    os << '@' << lab.ext;

    return os;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
