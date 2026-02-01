//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file accel/detail/MagneticFieldUtils.hh
//! \brief Utilities for magnetic field map creation
//---------------------------------------------------------------------------//
#pragma once

#include <G4Field.hh>
#include <corecel/Assert.hh>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"
#include "corecel/data/HyperslabIndexer.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Common field sampling setup and execution.
 *
 * This function encapsulates the shared pattern of:
 * 1. Getting the G4 field
 * 2. Setting up the HyperslabIndexer
 * 3. Sampling the field on a grid
 *
 * \param g4field Geant4 magnetic field class
 * \param field_data Output parameter array to store field values (must be
 *                   pre-allocated with size equal to the product of all dims)
 * \param dims Grid dimensions
 * \param calc_position Callable that computes position given
 *                           grid indices. Must have signature:
 *                           Array<G4double, 4>(size_type, size_type,
 *                           size_type) returning [x, y, z, 0] coordinates
 * \param convert_field Callable that converts field from G4 to
 *                        native units in the correct coordinate space. Must
 *                        have signature:
 *                        void(Array<G4double, 3> const& field,
 *                             Array<G4double, 4> const& pos,
 *                             real_type output[3])
 *                        taking G4 field [Bx, By, Bz], the position
 *                        [x, y, z, 0], and writing converted values to output
 *                        pointer
 */
template<typename PositionCalc, typename FieldConverter>
inline void setup_and_sample_field(G4Field const& g4field,
                                   real_type* field_data,
                                   Array<size_type, 4> const& dims,
                                   PositionCalc const& calc_position,
                                   FieldConverter const& convert_field)
{
    HyperslabIndexer const flat_index{dims};
    Array<G4double, 3> bfield;

    for (size_type i = 0; i < dims[0]; ++i)
    {
        for (size_type j = 0; j < dims[1]; ++j)
        {
            for (size_type k = 0; k < dims[2]; ++k)
            {
                // Calculate position for this grid point
                Array<G4double, 4> pos = calc_position(i, j, k);

                // Sample field at this position
                g4field.GetFieldValue(pos.data(), bfield.data());

                // Convert and store field values
                auto* cur_bfield = field_data + flat_index(i, j, k, 0);
                convert_field(bfield, pos, cur_bfield);
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
