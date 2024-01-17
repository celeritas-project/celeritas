//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/ImportedElementalMapLoader.hh
//---------------------------------------------------------------------------//
#pragma once

#include <map>

#include "corecel/Assert.hh"
#include "celeritas/phys/AtomicNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Load ImportT data, mapped by integers for each element.
 *
 * This is used for Seltzer-Berger, LivermorePE, and other data stored by
 * element in \c ImportData.
 */
template<class T>
struct ImportedElementalMapLoader
{
    std::map<int, T> const& tables;

    inline T operator()(AtomicNumber z) const;
};

//---------------------------------------------------------------------------//
// INLINE HELPER FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Helper function to construct an ImportedElementalMapLoader.
 */
template<class T>
inline ImportedElementalMapLoader<T>
make_imported_element_loader(std::map<int, T> const& data)
{
    return {data};
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with defaults.
 */
template<class T>
T ImportedElementalMapLoader<T>::operator()(AtomicNumber z) const
{
    CELER_EXPECT(z);
    auto iter = tables.find(z.unchecked_get());
    CELER_VALIDATE(iter != tables.end(),
                   << "missing imported data for Z=" << z.get());
    return iter->second;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
