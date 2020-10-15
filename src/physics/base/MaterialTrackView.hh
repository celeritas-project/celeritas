//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file MaterialTrackView.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "ElementView.hh"
#include "MaterialStatePointers.hh"
#include "MaterialParamsPointers.hh"
#include "Units.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Read/write view to the physical properties of a single particle track.
 *
 * These functions should be used in each physics Process or Interactor or
 * anything else that needs to access particle properties. Assume that all
 * these functions are expensive: when using them as accessors, locally store
 * the results rather than calling the function repeatedly. If any of the
 * calculations prove to be hot spots we will experiment with cacheing some of
 * the variables.
 */
class MaterialTrackView
{
  public:
    //@{
    //! Type aliases
    using Initializer_t = MaterialTrackState;
    //@}

  public:
    // Construct from "dynamic" state and "static" particle definitions
    inline CELER_FUNCTION
    MaterialTrackView(const MaterialParamsPointers& params,
                      const MaterialStatePointers&  states,
                      ThreadId                      id);

    // Initialize the particle
    inline CELER_FUNCTION MaterialTrackView&
                          operator=(const Initializer_t& other);

    // >>> DYNAMIC PROPERTIES (pure accessors, free)

    // Unique material identifier (index in cross section tables, etc.)
    CELER_FUNCTION MaterialDefId def_id() const { return state_.def_id; }

    // Material density
    CELER_FUNCTION real_type density() const
    {
        return this->material_def().density;
    }
    CELER_FUNCTION real_type number_density() const
    {
        return this->material_def().number_density;
    }
    CELER_FUNCTION real_type temperature() const
    {
        return this->material_def().temperature;
    }
    CELER_FUNCTION real_type electron_density() const
    {
        return this->material_def().electron_density;
    }
    CELER_FUNCTION real_type radiation_length_tsai() const
    {
        return this->material_def().radiation_length_tsai;
    }

    // >>> STATIC PROPERTIES

    CELER_FUNCTION size_type num_elements() const
    {
        return this->material_def().elements.size();
    }

    // Get from a global element ID
    ElementView get_element(ElementDefId el_id) const
    {
        return ElementView(params_, el_id);
    }

    // Get from an index into the elements of this material
    ElementView get_element(ElementComponentId id) const
    {
        REQUIRE(id < this->material_def().elements.size());
        const MatElementComponent& c = this->material_def().elements[id.get()];
        return ElementView(params_, c.element);
    }

  private:
    // >>> HELPER FUNCTIONS

    inline CELER_FUNCTION const MaterialDef& material_def() const;

  private:
    const MaterialParamsPointers& params_;
    MaterialTrackState&           state_;
};

//---------------------------------------------------------------------------//
} // namespace celeritas

#include "MaterialTrackView.i.hh"
