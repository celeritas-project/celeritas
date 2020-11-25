#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# File: scripts/dev/celeritas-gen.py
###############################################################################
"""Generate class file stubs for Celeritas.
"""
from __future__ import (division, absolute_import, print_function)
import os.path
import re
import subprocess
import sys
###############################################################################

CLIKE_TOP = '''\
//{modeline:-^75s}//
// Copyright {year} UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \\file {filename}
//---------------------------------------------------------------------------//
'''

HEADER_FILE = '''\
#pragma once

#include "base/Macros.hh"
#include "base/Types.hh"
#include "physics/base/Interaction.hh"
#include "physics/base/ParticleTrackView.hh"
#include "physics/base/SecondaryAllocatorView.hh"
#include "physics/base/Secondary.hh"
#include "physics/base/Units.hh"
#include "{name}Pointers.hh"

namespace {namespace}
{{
//---------------------------------------------------------------------------//
/*!
 * Brief class description.
 *
 * This is a model for XXXX process. Additional description
 *
 * \\note This performs the same sampling routine as in Geant4's
 * XXXX class, as documented in section XXX of the Geant4 Physics
 * Reference (release 10.6).
 */
class {name}
{{
  public:
    // Construct with shared and state data
    inline CELER_FUNCTION {name}(
            const {name}Pointers&    shared,
            const ParticleTrackView& particle,
            const Real3&             inc_direction,
            SecondaryAllocatorView&  allocate);


    // Sample an interaction with the given RNG
    template<class Engine>
    inline CELER_FUNCTION Interaction operator()(Engine& rng);

    // >>> COMMON PROPERTIES

    //! Minimum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy min_incident_energy()
    {{
        return units::MevEnergy{{0}}; // XXX
    }}

    //! Maximum incident energy for this model to be valid
    static CELER_CONSTEXPR_FUNCTION units::MevEnergy max_incident_energy()
    {{
        return units::MevEnergy{{0}}; // XXX
    }}

  private:
    // Shared constant physics properties
    const {name}Pointers& shared_;
    // Incident gamma energy
    const units::MevEnergy inc_energy_;
    // Incident direction
    const Real3& inc_direction_;
    // Allocate space for one or more secondary particles
    SecondaryAllocatorView& allocate_;
}};

//---------------------------------------------------------------------------//
}} // namespace {namespace}

#include "{name}.i.{hext}"
'''

INLINE_FILE = '''\

namespace {namespace}
{{
//---------------------------------------------------------------------------//
/*!
 * Construct with shared and state data.
 */
CELER_FUNCTION {name}::{name}(
        const {name}Pointers& shared,
        const ParticleTrackView& particle,
        const Real3&             inc_direction,
        SecondaryAllocatorView&  allocate)
    : shared_(shared)
    , inc_energy_(particle.energy().value())
    , inc_direction_(inc_direction)
    , allocate_(allocate)
{{
    REQUIRE(inc_energy_ >= this->min_incident_energy()
            && inc_energy_ <= this->max_incident_energy());
    REQUIRE(particle.def_id() == shared_.gamma_id); // XXX
}}

//---------------------------------------------------------------------------//
/*!
 * Sample using the XXX model.
 */
template<class Engine>
CELER_FUNCTION Interaction {name}::operator()(Engine& rng)
{{
    // Allocate space for XXX (electron, multiple particles, ...)
    Secondary* secondaries = this->allocate_(0); // XXX
    if (secondaries == nullptr)
    {{
        // Failed to allocate space for a secondary
        return Interaction::from_failure();
    }}

    // XXX sample
    (void)sizeof(rng);

    // Construct interaction for change to primary (incident) particle
    Interaction result;
    result.action      = Action::scattered; // XXX
    result.energy      = units::MevEnergy{{inc_energy_.value()}}; // XXX
    result.direction   = inc_direction_;
    result.secondaries = {{secondaries, 1}}; // XXX

    // Save outgoing secondary data
    secondaries[0].def_id = shared_.electron_id; // XXX
    secondaries[0].energy = units::MevEnergy{{0}}; // XXX
    secondaries[0].direction = {{0, 0, 0}}; // XXX

    return result;
}}

//---------------------------------------------------------------------------//
}}  // namespace {namespace}
'''

CODE_FILE = '''\
#include "{name}.{hext}"

namespace {namespace}
{{
//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
}} // namespace {namespace}
'''

TEST_HARNESS_FILE = '''\
#include "physics/em/{name}.{hext}"

#include "gtest/Main.hh"
#include "base/ArrayUtils.hh"
#include "base/Range.hh"
#include "physics/base/Units.hh"
#include "../InteractorHostTestBase.hh"
#include "../InteractionIO.hh"

using {namespace}::{name};
namespace pdg = celeritas::pdg;

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class {name}Test : public celeritas_test::InteractorHostTestBase
{{
    using Base = celeritas_test::InteractorHostTestBase;
  protected:
    void SetUp() override
    {{
        using celeritas::ParticleDef;
        using namespace celeritas::units;
        constexpr auto zero   = celeritas::zero_quantity();
        constexpr auto stable = ParticleDef::stable_decay_constant();

        // XXX Update these based on particles needed by interactor
        Base::set_particle_params(
            {{{{{{"electron", pdg::electron()}},
              {{MevMass{{0.5109989461}}, ElementaryCharge{{-1}}, stable}}}},
             {{{{"gamma", pdg::gamma()}}, {{zero, zero, stable}}}}}});
        const auto& params    = this->particle_params();
        pointers_.electron_id = params.find(pdg::electron());
        pointers_.gamma_id    = params.find(pdg::gamma());

        // Set default particle to incident XXX MeV photon
        this->set_inc_particle(pdg::gamma(), MevEnergy{{10}});
        this->set_inc_direction({{0, 0, 1}});
    }}

    void sanity_check(const Interaction& interaction) const
    {{
        ASSERT_TRUE(interaction);

        // Check change to parent track
        EXPECT_GT(this->particle_track().energy().value(),
                  interaction.energy.value());
        EXPECT_LT(0, interaction.energy.value());
        EXPECT_SOFT_EQ(1.0, celeritas::norm(interaction.direction));
        EXPECT_EQ(celeritas::Action::scattered, interaction.action);

        // XXX Check secondaries

        // Check conservation between primary and secondaries
        this->check_conservation(interaction);
    }}

  protected:
    celeritas::{name}Pointers pointers_;
}};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F({name}Test, basic)
{{
}}
'''

TEST_HEADER_FILE = '''
namespace celeritas_test
{{
using namespace {namespace};
//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct {capabbr}TestInput
{{
    int num_threads;
}};

//---------------------------------------------------------------------------//
//! Output results
struct {capabbr}TestOutput
{{
}};

//---------------------------------------------------------------------------//
//! Run on device and return results
{capabbr}TestOutput {lowabbr}_test({capabbr}TestInput);

//---------------------------------------------------------------------------//
}} // namespace celeritas_test
'''

TEST_CODE_FILE = '''\
#include "{name}.test.hh"

#include <thrust/device_vector.h>
#include "base/KernelParamCalculator.cuda.hh"

using thrust::raw_pointer_cast;

namespace celeritas_test
{{
//---------------------------------------------------------------------------//
// KERNELS
//---------------------------------------------------------------------------//

__global__ void {lowabbr}_test_kernel(unsigned int size)
{{
    auto local_thread_id = celeritas::KernelParamCalculator::thread_id();
    if (local_thread_id.get() >= size)
        return;
}}

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Run on device and return results
{capabbr}TestOutput {lowabbr}_test({capabbr}TestInput input)
{{
    celeritas::KernelParamCalculator calc_launch_params;
    auto params = calc_launch_params(input.num_threads);
    {lowabbr}_test_kernel<<<params.grid_size, params.block_size>>>(
        input.num_threads);

    {capabbr}TestOutput result;
    return result;
}}

//---------------------------------------------------------------------------//
}} // namespace celeritas_test
'''


CMAKE_TOP = '''\
#{modeline:-^77s}#
# Copyright {year} UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
'''

CMAKELISTS_FILE = '''\
#-----------------------------------------------------------------------------#

'''


CMAKE_FILE = '''\
#[=======================================================================[.rst:

{name}
-------------------

Description of overall module contents goes here.

.. command:: my_command_name

  Pass the given compiler-dependent warning flags to a library target::

    my_command_name(<target>
                    <INTERFACE|PUBLIC|PRIVATE>
                    LANGUAGE <lang> [<lang>...]
                    [CACHE_VARIABLE <name>])

  ``target``
    Name of the library target.

  ``scope``
    One of ``INTERFACE``, ``PUBLIC``, or ``PRIVATE``. ...

#]=======================================================================]

function(my_command_name)
endfunction()

'''

PYTHON_TOP = '''\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright {year} UT-Battelle, LLC and other Celeritas Developers.
# See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: (Apache-2.0 OR MIT)
'''

PYTHON_FILE = '''\
"""
"""

'''

YEAR = "2020"

TEMPLATES = {
    'hh': HEADER_FILE,
    'i.hh': INLINE_FILE,
    'cc': CODE_FILE,
    'cu': CODE_FILE,
    'cuh': HEADER_FILE,
    'test.cc': TEST_HARNESS_FILE,
    'test.cu': TEST_CODE_FILE,
    'test.hh': TEST_HEADER_FILE,
    'k.cuh': INLINE_FILE,
    'i.cuh': INLINE_FILE,
    't.cuh': INLINE_FILE,
    'cmake': CMAKE_FILE,
    'CMakeLists.txt': CMAKELISTS_FILE,
    'py': PYTHON_FILE,
}

LANG = {
    'h': "C++",
    'hh': "C++",
    'cc': "C++",
    'cu': "CUDA",
    'cuh': "CUDA",
    'cmake': "CMake",
    'CMakeLists.txt': "CMake",
    'py': "Python",
}

TOPS = {
    'C++': CLIKE_TOP,
    'CUDA': CLIKE_TOP,
    'CMake': CMAKE_TOP,
    'Python': PYTHON_TOP,
}

HEXT = {
    'C++': "hh",
    'CUDA': "cuh",
}

def generate(root, filename, namespace):
    if os.path.exists(filename):
        print("Skipping existing file " + filename)
        return

    relpath = os.path.relpath(filename, start=root)
    dirname = os.path.dirname(relpath)
    basename = os.path.basename(filename)
    (name, _, longext) = basename.partition('.')

    lang = None
    template = None
    ext = longext.split('.')[-1]
    for check_lang in [basename, longext, ext]:
        if not lang:
            lang = LANG.get(check_lang, None)
        if not template:
            template = TEMPLATES.get(check_lang, None)
    if not lang:
        print(f"No known language for '.{ext}' files")
    if not template:
        print(f"No configured template for '.{ext}' files")
    if not lang or not template:
        sys.exit(1)

    top = TOPS[lang]

    relpath = re.sub(r'^[./]+', '', relpath)
    capabbr = re.sub(r'[^A-Z]+', '', name)
    dirfromtest = re.sub(r'^test/', '', relpath)
    variables = {
        'longext': longext,
        'ext': ext,
        'hext': HEXT.get(lang, ext),
        'modeline': "-*-{}-*-".format(lang),
        'name': name,
        'namespace': namespace,
        'filename': filename,
        'basename': basename,
        'dirfromtest': dirfromtest,
        'capabbr': capabbr,
        'lowabbr': capabbr.lower(),
        'year': YEAR,
        }
    with open(filename, 'w') as f:
        f.write((top + template).format(**variables))

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename', nargs='+',
                        help='file names to generate')
    parser.add_argument('--basedir',
                        help='root source directory for file naming')
    parser.add_argument('--namespace', '-n',
                        default='celeritas',
                        help='root source directory for file naming')
    args = parser.parse_args()
    basedir = args.basedir or os.path.join(
            subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
                .decode().strip(),
            'src')
    for fn in args.filename:
        generate(basedir, fn, args.namespace)

#-----------------------------------------------------------------------------#
if __name__ == '__main__':
    main()

###############################################################################
# end of scripts/dev/celeritas-gen.py
###############################################################################
