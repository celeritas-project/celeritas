//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/em/params/WentzelOKVIParams.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/data/CollectionMirror.hh"
#include "corecel/data/ParamsDataInterface.hh"
#include "celeritas/em/data/WentzelOKVIData.hh"
#include "celeritas/phys/AtomicNumber.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class MaterialParams;
class ParticleParams;
struct ImportData;

//---------------------------------------------------------------------------//
/*!
 * Construct and store shared Coulomb and multiple scattering data.
 *
 * This data is used by both the single Coulomb scattering and Wentzel VI
 * multiple scattering models.
 */
class WentzelOKVIParams final : public ParamsDataInterface<WentzelOKVIData>
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstMaterials = std::shared_ptr<MaterialParams const>;
    using SPConstParticles = std::shared_ptr<ParticleParams const>;
    //!@}

  public:
    struct Options
    {
        //! Use combined single and multiple scattering
        bool is_combined{true};
        //! Polar angle limit between single and multiple scattering
        real_type polar_angle_limit{constants::pi};
        //! Factor for dynamic computation of angular limit between SS and MSC
        real_type angle_limit_factor{1};
        //! User defined screening factor
        real_type screening_factor{1};
        //! Nuclear form factor model
        NuclearFormFactorType form_factor{NuclearFormFactorType::exponential};
    };

  public:
    // Construct if Wentzel VI or Coulomb is present, else return nullptr
    static std::shared_ptr<WentzelOKVIParams>
    from_import(ImportData const& data,
                SPConstMaterials materials,
                SPConstParticles particles);

    // Construct from material data and options
    WentzelOKVIParams(SPConstMaterials materials,
                      SPConstParticles particles,
                      Options options);

    //! Access Wentzel OK&VI data on the host
    HostRef const& host_ref() const final { return data_.host_ref(); }

    //! Access Wentzel OK&VI data on the device
    DeviceRef const& device_ref() const final { return data_.device_ref(); }

  private:
    //// TYPES ////

    using CoeffMat = MottElementData::MottCoeffMatrix;

    //// DATA ////

    // Host/device storage and reference
    CollectionMirror<WentzelOKVIData> data_;

    //// HELPER METHODS ////

    // Construct per-element data (loads Mott coefficients)
    void build_data(HostVal<WentzelOKVIData>& host_data,
                    MaterialParams const& materials);

    // Retrieve matrix of interpolated Mott electron coefficients
    static CoeffMat get_electron_mott_coeffs(AtomicNumber z);

    // Retrieve matrix of interpolated Mott positron coefficients
    static CoeffMat get_positron_mott_coeffs(AtomicNumber z);
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
