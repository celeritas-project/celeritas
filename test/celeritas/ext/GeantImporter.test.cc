//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/ext/GeantImporter.test.cc
//---------------------------------------------------------------------------//
#include "celeritas/ext/GeantImporter.hh"

#include "corecel/Config.hh"

#include "corecel/ScopedLogStorer.hh"
#include "corecel/io/Logger.hh"
#include "corecel/io/Repr.hh"
#include "corecel/sys/Version.hh"
#include "geocel/UnitUtils.hh"
#include "celeritas/GeantTestBase.hh"
#include "celeritas/Types.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/ext/GeantPhysicsOptionsIO.json.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/PDGNumber.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// Helper functions
namespace
{

using namespace celeritas::units;

template<class Iter>
std::vector<std::string> to_vec_string(Iter iter, Iter end)
{
    std::vector<std::string> result;
    for (; iter != end; ++iter)
    {
        result.push_back(to_cstring(*iter));
    }
    return result;
}

real_type to_sec(real_type v)
{
    return native_value_to<RealQuantity<Second>>(v).value();
}

auto const geant4_version = Version::from_string(cmake::geant4_version);
}  // namespace

//---------------------------------------------------------------------------//
// TEST HARNESS
//---------------------------------------------------------------------------//

class GeantImporterTest : public GeantTestBase
{
  protected:
    using DataSelection = GeantImportDataSelection;
    using VecModelMaterial = std::vector<ImportModelMaterial>;

    struct ImportSummary
    {
        std::vector<std::string> particles;
        std::vector<std::string> processes;
        std::vector<std::string> models;

        void print_expected() const;
    };

    struct ImportXsSummary
    {
        std::vector<size_type> size;  //!< Number of micro XS in each material
        std::vector<real_type> energy;
        std::vector<real_type> xs;

        void print_expected() const;
    };

    ImportSummary summarize(ImportData const& data) const;
    ImportXsSummary summarize(VecModelMaterial const& xs) const;

    // Import data potentially with different selection options
    GeantImportDataSelection build_import_data_selection() const final
    {
        return selection_;
    }

    ImportProcess const&
    find_process(PDGNumber pdg, ImportProcessClass ipc) const
    {
        auto const& processes = this->imported_data().processes;
        auto result = std::find_if(processes.begin(),
                                   processes.end(),
                                   [pdg, ipc](ImportProcess const& proc) {
                                       return PDGNumber{proc.particle_pdg}
                                                  == pdg
                                              && proc.process_class == ipc;
                                   });
        CELER_VALIDATE(result != processes.end(),
                       << "missing process " << to_cstring(ipc)
                       << " for particle PDG=" << pdg.get());
        return *result;
    }

    ImportMscModel const&
    find_msc_model(PDGNumber pdg, ImportModelClass imc) const
    {
        auto const& models = this->imported_data().msc_models;
        auto result = std::find_if(
            models.begin(), models.end(), [pdg, imc](ImportMscModel const& m) {
                return PDGNumber{m.particle_pdg} == pdg && m.model_class == imc;
            });
        CELER_VALIDATE(result != models.end(),
                       << "missing model " << to_cstring(imc)
                       << " for particle PDG=" << pdg.get());
        return *result;
    }

    real_type comparison_tolerance() const
    {
        if (geant4_version != Version(11, 0, 3))
        {
            // Some values change substantially between geant versions
            return 5e-3;
        }
        if (CELERITAS_REAL_TYPE != CELERITAS_REAL_TYPE_DOUBLE)
        {
            // Single-precision unit constants cause single-precision
            // differences from reference
            return 1e-6;
        }
        return 1e-12;
    }

  protected:
    GeantImportDataSelection selection_{};
};

//---------------------------------------------------------------------------//
auto GeantImporterTest::summarize(ImportData const& data) const -> ImportSummary
{
    ImportSummary s;
    for (auto const& p : data.particles)
    {
        s.particles.push_back(p.name);
    }

    // Create sorted unique set of process and model names inserted
    std::set<ImportProcessClass> pclass;
    std::set<ImportModelClass> mclass;
    for (auto const& p : data.processes)
    {
        pclass.insert(p.process_class);
        for (auto const& m : p.models)
        {
            mclass.insert(m.model_class);
        }
    }
    for (auto const& m : data.msc_models)
    {
        mclass.insert(m.model_class);
    }
    s.processes = to_vec_string(pclass.begin(), pclass.end());
    s.models = to_vec_string(mclass.begin(), mclass.end());
    return s;
}

void GeantImporterTest::ImportSummary::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
            "static const char* expected_particles[] = "
         << repr(this->particles) << ";\n"
         << "EXPECT_VEC_EQ(expected_particles, summary.particles);\n"
            "static const char* expected_processes[] = "
         << repr(this->processes) << ";\n"
         << "EXPECT_VEC_EQ(expected_processes, summary.processes);\n"
            "static const char* expected_models[] = "
         << repr(this->models) << ";\n"
         << "EXPECT_VEC_EQ(expected_models, summary.models);\n"
            "/*** END CODE ***/\n";
}

auto GeantImporterTest::summarize(VecModelMaterial const& materials) const
    -> ImportXsSummary
{
    ImportXsSummary result;
    for (auto const& mat : materials)
    {
        result.size.push_back(
            mat.micro_xs.empty() ? 0 : mat.micro_xs.front().y.size());
        result.energy.push_back(mat.energy[Bound::lo]);
        result.energy.push_back(mat.energy[Bound::hi]);
    }

    // Skip export of first material, which is usually vacuum
    std::size_t mat_idx = 0;
    for (auto const& xs_vec : materials[mat_idx].micro_xs)
    {
        EXPECT_EQ(result.size[mat_idx], xs_vec.y.size());
    }
    ++mat_idx;

    for (; mat_idx < materials.size(); ++mat_idx)
    {
        for (auto const& xs_vec : materials[mat_idx].micro_xs)
        {
            EXPECT_EQ(result.size[mat_idx], xs_vec.y.size());
            result.xs.push_back(xs_vec.y.front() / barn);
            result.xs.push_back(xs_vec.y.back() / barn);
        }
    }
    return result;
}

void GeantImporterTest::ImportXsSummary::print_expected() const
{
    cout << "/*** ADD THE FOLLOWING UNIT TEST CODE ***/\n"
         << "static size_type const expected_size[] = " << repr(this->size)
         << ";\n"
         << "EXPECT_VEC_EQ(expected_size, result.size);\n"
         << "static real_type const expected_e[] = " << repr(this->energy)
         << ";\n"
         << "EXPECT_VEC_SOFT_EQ(expected_e, result.energy);\n"
         << "static real_type const expected_xs[] = " << repr(this->xs)
         << ";\n"
         << "EXPECT_VEC_SOFT_EQ(expected_xs, result.xs);\n"
         << "/*** END CODE ***/\n";
}

//---------------------------------------------------------------------------//
class LarSphere : public GeantImporterTest
{
  protected:
    std::string_view gdml_basename() const override { return "lar-sphere"sv; }

    GeantPhysicsOptions build_geant_options() const override
    {
        auto opts = GeantImporterTest::build_geant_options();
        opts.optical.emplace();
        return opts;
    }
};

//---------------------------------------------------------------------------//
class LarSphereExtramat : public LarSphere
{
  protected:
    std::string_view gdml_basename() const override
    {
        return "lar-sphere-extramat"sv;
    }
};

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

TEST_F(LarSphere, optical)
{
    ScopedLogStorer scoped_log{&celeritas::world_logger(), LogLevel::info};
    auto&& imported = this->imported_data();
    ASSERT_EQ(1, imported.optical_materials.size());
    ASSERT_EQ(3, imported.geo_materials.size());
    ASSERT_EQ(2, imported.phys_materials.size());

    // First material is vacuum, no optical properties
    ASSERT_EQ(0, imported.phys_materials[0].geo_material_id);
    EXPECT_EQ("vacuum", imported.geo_materials[0].name);
    EXPECT_EQ(ImportPhysMaterial::unspecified,
              imported.phys_materials[0].optical_material_id);

    // Second material is liquid argon
    ASSERT_EQ(1, imported.phys_materials[1].geo_material_id);
    EXPECT_EQ("lAr", imported.geo_materials[1].name);
    ASSERT_EQ(0, imported.phys_materials[1].optical_material_id);

    // Most optical properties in the geometry are pulled from the Geant4
    // example examples/advanced/CaTS/gdml/LArTPC.gdml

    // Check scintillation optical properties
    auto const& optical = imported.optical_materials[0];
    auto const& scint = optical.scintillation;
    EXPECT_TRUE(scint);

    // Material scintillation
    constexpr auto tol = SoftEqual<real_type>{}.rel();
    EXPECT_REAL_EQ(1, scint.resolution_scale);
    EXPECT_REAL_EQ(5000, scint.material.yield_per_energy);
    EXPECT_EQ(3, scint.material.components.size());
    std::vector<double> components;
    for (auto const& comp : scint.material.components)
    {
        components.push_back(comp.yield_frac);
        components.push_back(to_cm(comp.gauss.lambda_mean));
        components.push_back(to_cm(comp.gauss.lambda_sigma));
        components.push_back(to_sec(comp.rise_time));
        components.push_back(to_sec(comp.fall_time));
    }
    static double const expected_components[] = {
        3,
        1.28e-05,
        1e-06,
        1e-08,
        6e-09,
        1,
        1.28e-05,
        1e-06,
        1e-08,
        1.5e-06,
        1,
        0,
        0,
        1e-08,
        3e-06,
    };
    EXPECT_VEC_NEAR(expected_components, components, tol);

    // Particle scintillation
    EXPECT_EQ(6, scint.particles.size());
    std::vector<int> pdgs;
    std::vector<double> yield_vecs;
    std::vector<size_t> comp_sizes;
    std::vector<double> comp_y, comp_lm, comp_ls, comp_rt, comp_ft;
    for (auto const& iter : scint.particles)
    {
        pdgs.push_back(iter.first);
        auto const& part = iter.second;
        for (auto i : range(part.yield_vector.x.size()))
        {
            yield_vecs.push_back(part.yield_vector.x[i]);
            yield_vecs.push_back(part.yield_vector.y[i]);
        }
        comp_sizes.push_back(part.components.size());
        for (auto comp : part.components)
        {
            comp_y.push_back(comp.yield_frac);
            comp_lm.push_back(to_cm(comp.gauss.lambda_mean));
            comp_ls.push_back(to_cm(comp.gauss.lambda_sigma));
            comp_rt.push_back(to_sec(comp.rise_time));
            comp_ft.push_back(to_sec(comp.fall_time));
        }
    }
    static int const expected_pdgs[]
        = {11, 90, 2212, 1000010020, 1000010030, 1000020040};
    static double const expected_yield_vecs[] = {
        1e-06, 3750, 6, 5000,  // electron
        1e-06, 2000, 6, 4000,  // ion
        1e-06, 2500, 6, 4200,  // proton
        1e-06, 1200, 6, 3000,  // deuteron
        1e-06, 1500, 6, 3500,  // triton
        1e-06, 1700, 6, 3700  // alpha
    };
    EXPECT_VEC_EQ(expected_pdgs, pdgs);
    EXPECT_VEC_EQ(expected_yield_vecs, yield_vecs);

    // The electron has one component, the rest has no components
    static unsigned long const expected_comp_sizes[]
        = {1ul, 0ul, 0ul, 0ul, 0ul, 0ul};
    EXPECT_VEC_EQ(expected_comp_sizes, comp_sizes);

    // Electron component data
    static double const expected_comp_y[] = {4000};
    static double const expected_comp_lm[] = {1e-05};
    static double const expected_comp_ls[] = {1e-06};
    static double const expected_comp_rt[] = {1.5e-08};
    static double const expected_comp_ft[] = {5e-09};

    EXPECT_VEC_EQ(expected_comp_y, expected_comp_y);
    EXPECT_VEC_EQ(expected_comp_lm, expected_comp_lm);
    EXPECT_VEC_EQ(expected_comp_ls, expected_comp_ls);
    EXPECT_VEC_EQ(expected_comp_rt, expected_comp_rt);
    EXPECT_VEC_EQ(expected_comp_ft, expected_comp_ft);

    auto& bulk = imported.optical_physics.bulk;
    // Check Rayleigh optical properties
    auto const& rayleigh_mfp = bulk.rayleigh.materials.at(OptMatId{0}).mfp;
    EXPECT_EQ(11, rayleigh_mfp.x.size());
    EXPECT_DOUBLE_EQ(1.55e-06, rayleigh_mfp.x.front());
    EXPECT_DOUBLE_EQ(1.55e-05, rayleigh_mfp.x.back());
    EXPECT_REAL_EQ(32142.9, to_cm(rayleigh_mfp.y.front()));
    EXPECT_REAL_EQ(54.6429, to_cm(rayleigh_mfp.y.back()));

    // Check absorption optical properties
    auto const& absorption_mfp = bulk.absorption.materials.at(OptMatId{0}).mfp;
    EXPECT_EQ(2, absorption_mfp.x.size());
    EXPECT_DOUBLE_EQ(1.3778e-06, absorption_mfp.x.front());
    EXPECT_DOUBLE_EQ(1.55e-05, absorption_mfp.x.back());
    EXPECT_REAL_EQ(86.4473, to_cm(absorption_mfp.y.front()));
    EXPECT_REAL_EQ(0.000296154, to_cm(absorption_mfp.y.back()));

    {
        // Check WLS optical properties
        auto const& mat = bulk.wls.materials.at(OptMatId{0});
        auto const& mfp = mat.mfp;
        EXPECT_EQ(2, mfp.x.size());
        EXPECT_EQ(mfp.x.size(), mfp.y.size());

        EXPECT_TRUE(mat);
        EXPECT_SOFT_EQ(0.456, mat.mean_num_photons);
        EXPECT_SOFT_EQ(6e-9, to_sec(mat.time_constant));

        std::vector<double> abslen_grid, comp_grid;
        for (auto i : range(mfp.x.size()))
        {
            abslen_grid.push_back(mfp.x[i]);
            abslen_grid.push_back(to_cm(mfp.y[i]));
            comp_grid.push_back(mat.component.x[i]);
            comp_grid.push_back(mat.component.y[i]);
        }

        static real_type const expected_abslen_grid[]
            = {1.3778e-06, 0.1, 1.55e-05, 0.01};
        static double const expected_comp_grid[]
            = {1.3778e-06, 0.1, 1e-05, 0.9};
        EXPECT_VEC_SOFT_EQ(expected_abslen_grid, abslen_grid);
        EXPECT_VEC_SOFT_EQ(expected_comp_grid, comp_grid);
    }
    {
        // Check WLS2 optical properties
        auto const& mat = bulk.wls2.materials.at(OptMatId{0});
        auto const& mfp = mat.mfp;
        EXPECT_EQ(2, mfp.x.size());
        EXPECT_EQ(mfp.x.size(), mfp.y.size());

        EXPECT_TRUE(mat);
        EXPECT_REAL_EQ(0.123, mat.mean_num_photons);
        EXPECT_REAL_EQ(6e-9, to_sec(mat.time_constant));

        std::vector<double> abslen_grid, comp_grid;
        for (auto i : range(mfp.x.size()))
        {
            abslen_grid.push_back(mfp.x[i]);
            abslen_grid.push_back(to_cm(mfp.y[i]));
            comp_grid.push_back(mat.component.x[i]);
            comp_grid.push_back(mat.component.y[i]);
        }

        static double const expected_abslen_grid[]
            = {1.3778e-06, 0.1, 1.55e-05, 0.01};
        static double const expected_comp_grid[]
            = {1.771e-06, 0.3, 2.484e-06, 0.8};
        EXPECT_VEC_NEAR(
            expected_abslen_grid, abslen_grid, this->comparison_tolerance());
        EXPECT_VEC_SOFT_EQ(expected_comp_grid, comp_grid);
    }

    // Check common optical properties
    // Refractive index data in the geometry comes from the refractive index
    // database https://refractiveindex.info and was calculating using the
    // methods described in: E. Grace, A. Butcher, J.  Monroe, J. A. Nikkel.
    // Index of refraction, Rayleigh scattering length, and Sellmeier
    // coefficients in solid and liquid argon and xenon, Nucl.  Instr. Meth.
    // Phys. Res. A 867, 204-208 (2017)
    auto const& properties = optical.properties;
    EXPECT_TRUE(properties);
    EXPECT_EQ(101, properties.refractive_index.x.size());
    EXPECT_DOUBLE_EQ(1.8785e-06, properties.refractive_index.x.front());
    EXPECT_DOUBLE_EQ(1.0597e-05, properties.refractive_index.x.back());
    EXPECT_DOUBLE_EQ(1.2221243542166, properties.refractive_index.y.front());
    EXPECT_DOUBLE_EQ(1.6167515615703, properties.refractive_index.y.back());
}

TEST_F(LarSphereExtramat, optical)
{
    auto&& imported = this->imported_data();
    ASSERT_EQ(1, imported.optical_materials.size());
    ASSERT_EQ(3, imported.geo_materials.size());
    ASSERT_EQ(2, imported.phys_materials.size());

    // First material is vacuum, no optical properties
    ASSERT_EQ(0, imported.phys_materials[0].geo_material_id);
    EXPECT_EQ("vacuum", imported.geo_materials[0].name);
    EXPECT_EQ(ImportPhysMaterial::unspecified,
              imported.phys_materials[0].optical_material_id);

    // Second material is liquid argon
    ASSERT_EQ(1, imported.phys_materials[1].geo_material_id);
    EXPECT_EQ("lAr", imported.geo_materials[1].name);
    ASSERT_EQ(0, imported.phys_materials[1].optical_material_id);

    // Check scintillation, WLS, and WLS2 optical properties
    auto const& optical = imported.optical_materials[0];
    EXPECT_FALSE(optical.scintillation);
    auto const& bulk = imported.optical_physics.bulk;
    EXPECT_FALSE(bulk.wls.materials.count(OptMatId{0}));
    EXPECT_FALSE(bulk.wls2.materials.count(OptMatId{0}));

    // Check Rayleigh optical properties
    auto const& rayleigh_mfp = bulk.rayleigh.materials.at(OptMatId{0}).mfp;
    EXPECT_EQ(2, rayleigh_mfp.x.size());
    EXPECT_DOUBLE_EQ(1.55e-06, rayleigh_mfp.x.front());
    EXPECT_DOUBLE_EQ(1.55e-05, rayleigh_mfp.x.back());
    EXPECT_REAL_EQ(32142.9, to_cm(rayleigh_mfp.y.front()));
    EXPECT_REAL_EQ(54.6429, to_cm(rayleigh_mfp.y.back()));

    // Check common optical properties
    // Refractive index data in the geometry comes from the refractive index
    // database https://refractiveindex.info and was calculating using the
    // methods described in: E. Grace, A. Butcher, J.  Monroe, J. A. Nikkel.
    // Index of refraction, Rayleigh scattering length, and Sellmeier
    // coefficients in solid and liquid argon and xenon, Nucl.  Instr. Meth.
    // Phys. Res. A 867, 204-208 (2017)
    auto const& properties = optical.properties;
    EXPECT_TRUE(properties);
    EXPECT_EQ(2, properties.refractive_index.x.size());
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
