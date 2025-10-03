//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/surface/model/DielectricInteractionData.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/cont/EnumArray.hh"
#include "corecel/data/Collection.hh"
#include "corecel/grid/NonuniformGridData.hh"
#include "celeritas/Quantities.hh"
#include "celeritas/grid/NonuniformGridCalculator.hh"
#include "celeritas/optical/Types.hh"

namespace celeritas
{
namespace optical
{
//---------------------------------------------------------------------------//
// TYPE ALIASES
//---------------------------------------------------------------------------//
// Dielectric interface type
enum class DielectricInterface : bool
{
    metal = true,
    dielectric = false,
};

// Supported reflection modes in UNIFIED model
enum class UnifiedReflectionMode
{
    specular_spike,
    specular_lobe,
    back_scattering,
    diffuse_lambertian,
    size_
};

using UnifiedModeProbs = EnumArray<UnifiedReflectionMode, real_type>;

//---------------------------------------------------------------------------//
// CLASSES
//---------------------------------------------------------------------------//
/*!
 * Data for the dielectric model denoting which interfaces are
 * dielectric-dielectric and dielectric-metal.
 */
template<Ownership W, MemSpace M>
struct DielectricData
{
    //!@{
    //! \name Type aliases
    template<class T>
    using SurfaceItems = Collection<T, W, M, SubModelId>;
    //!@}

    SurfaceItems<DielectricInterface> interface;

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !interface.empty();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    DielectricData<W, M>& operator=(DielectricData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        interface = other.interface;
        CELER_ENSURE(*this);
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Physics grids for the UNIFIED reflection model.
 */
template<Ownership W, MemSpace M>
struct UnifiedReflectionData
{
    //!@{
    //! \name Type aliases
    using SurfaceGrids = Collection<NonuniformGridRecord, W, M, SubModelId>;

    template<class T>
    using Items = Collection<T, W, M>;
    //!@}

    SurfaceGrids specular_spike;
    SurfaceGrids specular_lobe;
    SurfaceGrids back_scattering;

    //! Backend storage
    Items<real_type> reals;

    //! Whether data is assigned and valid
    explicit CELER_FUNCTION operator bool() const
    {
        return !specular_spike.empty()
               && specular_spike.size() == specular_lobe.size()
               && specular_spike.size() == back_scattering.size();
    }

    //! Assign from another set of data
    template<Ownership W2, MemSpace M2>
    UnifiedReflectionData<W, M>&
    operator=(UnifiedReflectionData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        specular_spike = other.specular_spike;
        specular_lobe = other.specular_lobe;
        back_scattering = other.back_scattering;
        reals = other.reals;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * A view into UNIFIED model grids to calculate reflection mode probabilities.
 */
class UnifiedReflectionView
{
  public:
    //!@{
    //! \name Type aliases
    using DataRef = NativeCRef<UnifiedReflectionData>;
    using Energy = units::MevEnergy;
    //!@}

  public:
    // Construct from data and a surface
    explicit inline CELER_FUNCTION
    UnifiedReflectionView(DataRef const&, SubModelId);

    // Calculate probability for each reflection mode
    inline CELER_FUNCTION UnifiedModeProbs operator()(Energy energy) const;

  private:
    DataRef const& data_;
    SubModelId surface_;

    // Calculate probability for a single reflection mode
    inline CELER_FUNCTION real_type calc_probability(NonuniformGridRecord const&,
                                                     Energy) const;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct view from reflection data and a surface.
 */
CELER_FUNCTION
UnifiedReflectionView::UnifiedReflectionView(DataRef const& data,
                                             SubModelId surface)
    : data_(data), surface_(surface)
{
    CELER_EXPECT(surface < data_.specular_spike.size());
}

//---------------------------------------------------------------------------//
/*!
 * Calculate probability for each reflection mode.
 *
 * Only the specular spike, specular lobe, and back-scattering probabilities
 * are defined as grids in the data. The diffuse Lambertian mode is the
 * remaining probability.
 */
CELER_FUNCTION auto UnifiedReflectionView::operator()(Energy energy) const
    -> UnifiedModeProbs
{
    UnifiedModeProbs probs;

    probs[UnifiedReflectionMode::specular_spike]
        = this->calc_probability(data_.specular_spike[surface_], energy);

    probs[UnifiedReflectionMode::specular_lobe]
        = this->calc_probability(data_.specular_lobe[surface_], energy);

    probs[UnifiedReflectionMode::back_scattering]
        = this->calc_probability(data_.back_scattering[surface_], energy);

    probs[UnifiedReflectionMode::diffuse_lambertian]
        = 1
          - (probs[UnifiedReflectionMode::specular_spike]
             + probs[UnifiedReflectionMode::specular_lobe]
             + probs[UnifiedReflectionMode::back_scattering]);

    CELER_ENSURE(probs[UnifiedReflectionMode::diffuse_lambertian] >= 0);

    return probs;
}

//---------------------------------------------------------------------------//
/*!
 * Calculate probability for a single reflection mode.
 */
CELER_FUNCTION real_type UnifiedReflectionView::calc_probability(
    NonuniformGridRecord const& grid, Energy energy) const
{
    NonuniformGridCalculator calc{grid, data_.reals};
    real_type result = calc(value_as<Energy>(energy));
    CELER_ENSURE(result >= 0 && result <= 1);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
