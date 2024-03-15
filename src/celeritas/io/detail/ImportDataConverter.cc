//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/io/detail/ImportDataConverter.cc
//---------------------------------------------------------------------------//
#include "ImportDataConverter.hh"

#include "corecel/Assert.hh"
#include "celeritas/UnitTypes.hh"

#include "../ImportData.hh"
#include "../ImportUnits.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Construct with a unit system.
 */
ImportDataConverter::ImportDataConverter(UnitSystem usys) : usys_{usys}
{
    len_ = native_value_from(usys_, ImportUnits::len);
    numdens_ = native_value_from(usys_, ImportUnits::inv_len_cb);
    time_ = native_value_from(usys_, ImportUnits::time);
    xs_ = native_value_from(usys_, ImportModelMaterial::xs_units);
    inv_pressure_ = native_value_from(usys_, ImportUnits::len_time_sq_per_mass);
}

//---------------------------------------------------------------------------//
void ImportDataConverter::operator()(ImportData* data)
{
    for (auto& m : data->materials)
    {
        (*this)(&m);
    }

    for (auto& m : data->optical)
    {
        (*this)(&m.second);
    }

    for (auto& p : data->processes)
    {
        (*this)(&p);
    }

    for (auto& m : data->msc_models)
    {
        (*this)(&m);
    }

    (*this)(&data->em_params);

    data->units = units::NativeTraits::label();
}

//---------------------------------------------------------------------------//
void ImportDataConverter::operator()(ImportElement* data)
{
    CELER_EXPECT(data);
}

//---------------------------------------------------------------------------//
void ImportDataConverter::operator()(ImportEmParameters* data)
{
    CELER_EXPECT(data);
    data->msc_lambda_limit *= len_;
}

//---------------------------------------------------------------------------//
void ImportDataConverter::operator()(ImportMaterial* data)
{
    CELER_EXPECT(data);

    data->number_density *= numdens_;

    for (auto& [pdg, cut] : data->pdg_cutoffs)
    {
        cut.range *= len_;
    }
}

//---------------------------------------------------------------------------//
void ImportDataConverter::operator()(ImportOpticalMaterial* data)
{
    CELER_EXPECT(data);

    for (auto& comp : data->scintillation.material.components)
    {
        comp.lambda_mean *= len_;
        comp.lambda_sigma *= len_;
        comp.rise_time *= time_;
        comp.fall_time *= time_;
    }
    for (auto& iter : data->scintillation.particles)
    {
        for (auto& comp : iter.second.components)
        {
            comp.lambda_mean *= len_;
            comp.lambda_sigma *= len_;
            comp.rise_time *= time_;
            comp.fall_time *= time_;
        }
    }
    for (auto& mfp : data->rayleigh.mfp.y)
    {
        mfp *= len_;
    }
    data->rayleigh.compressibility *= inv_pressure_;
    for (auto& abs_len : data->absorption.absorption_length.y)
    {
        abs_len *= len_;
    }
}

//---------------------------------------------------------------------------//
void ImportDataConverter::operator()(ImportModelMaterial* data)
{
    CELER_EXPECT(data);

    for (auto& xs_vec : data->micro_xs)
    {
        for (double& xs : xs_vec)
        {
            xs *= xs_;
        }
    }
}

//---------------------------------------------------------------------------//
void ImportDataConverter::operator()(ImportModel* data)
{
    CELER_EXPECT(data);

    for (auto& mm : data->materials)
    {
        (*this)(&mm);
    }
}

//---------------------------------------------------------------------------//
void ImportDataConverter::operator()(ImportMscModel* data)
{
    CELER_EXPECT(data);

    (*this)(&data->xs_table);
}

//---------------------------------------------------------------------------//
void ImportDataConverter::operator()(ImportParticle* data)
{
    CELER_EXPECT(data);

    data->lifetime *= time_;
}

//---------------------------------------------------------------------------//
void ImportDataConverter::operator()(ImportPhysicsTable* data)
{
    CELER_EXPECT(data);

    if (double const units = native_value_from(usys_, data->x_units);
        units != 1)
    {
        for (auto& v : data->physics_vectors)
        {
            for (auto& xval : v.x)
            {
                xval *= units;
            }
        }
    }

    if (double const units = native_value_from(usys_, data->y_units);
        units != 1)
    {
        for (auto& v : data->physics_vectors)
        {
            for (auto& yval : v.y)
            {
                yval *= units;
            }
        }
    }
}

//---------------------------------------------------------------------------//
void ImportDataConverter::operator()(ImportProcess* data)
{
    CELER_EXPECT(data);

    for (auto& m : data->models)
    {
        (*this)(&m);
    }

    for (auto& t : data->tables)
    {
        (*this)(&t);
    }
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
