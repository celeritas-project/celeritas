//----------------------------------*-C++-*----------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/ddceler-compare.cc
//! \brief Compare DD4hep simulation data between two configurations
//---------------------------------------------------------------------------//
/*!
 * Usage:
 * $ ddceler-compare <file1.root> <file2.root>
 *
 * DD4hep simulation output comparison for validation studies.
 *
 * Validation tests performed (6 total):
 * - Initial distributions (pt, eta, phi): KS test (p-value > 0.05)
 * - Calorimeter energy: Total energy difference < 3%
 * - Shower profiles (r-z, phi-z): Percentiles within 3%
 */
//---------------------------------------------------------------------------//
// Standard library
#include <cstdlib>
#include <iostream>
#include <vector>

// ROOT headers
#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TMath.h>
#include <TPad.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TText.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

// DD4hep headers
#include "DDG4/Geant4Data.h"
#include "DDG4/Geant4Particle.h"

//---------------------------------------------------------------------------//
//! Constants
//---------------------------------------------------------------------------//
static int const n_bins = 50;
static TString const commit_hash = "";

//---------------------------------------------------------------------------//
//! Statistical comparison metrics for 1D histograms
//---------------------------------------------------------------------------//
struct ComparisonMetrics
{
    double ks_prob;
    double mean_z_score;
    double rms_rel_diff;
};

ComparisonMetrics calculate_1D_metrics(TH1D* h1, TH1D* h2)
{
    ComparisonMetrics metrics;

    // Kolmogorov-Smirnov test
    metrics.ks_prob = h1->KolmogorovTest(h2);

    // Z-score for mean comparison
    double mean1 = h1->GetMean();
    double mean2 = h2->GetMean();
    double rms1 = h1->GetRMS();
    double rms2 = h2->GetRMS();
    double n1 = h1->GetEntries();
    double n2 = h2->GetEntries();

    // Standard errors
    double se1 = (n1 > 0) ? rms1 / TMath::Sqrt(n1) : 0;
    double se2 = (n2 > 0) ? rms2 / TMath::Sqrt(n2) : 0;
    double se_combined = TMath::Sqrt(se1 * se1 + se2 * se2);

    metrics.mean_z_score = (se_combined > 0) ? (mean1 - mean2) / se_combined
                                             : 0;

    // RMS relative difference
    metrics.rms_rel_diff = (rms1 != 0) ? (rms1 - rms2) / rms1 * 100 : 0;

    return metrics;
}

bool print_1D_metrics(TString plot_type, ComparisonMetrics const& metrics)
{
    // For initial distributions: validate with KS test
    if (plot_type == "pt" || plot_type == "eta" || plot_type == "phi")
    {
        bool passed = metrics.ks_prob > 0.05;
        std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] " << plot_type
                  << ": KS p-value = " << metrics.ks_prob
                  << " (threshold: >0.05)" << std::endl;
        return passed;
    }

    // For energy: no validation here (done separately with total energy)
    return true;
}

//---------------------------------------------------------------------------//
//! Shower profile metrics for physics-based comparison
//---------------------------------------------------------------------------//
struct ShowerMetrics
{
    // Penetration depth (longitudinal)
    double z_10_percentile;  // 10% of hits have z < this
    double z_50_percentile;  // Median z
    double z_90_percentile;  // 90% of hits have z < this
    double z_mean;
    double z_rms;

    // Radial spread (transverse)
    double r_mean;
    double r_rms;
    double r_10_percentile;
    double r_50_percentile;
    double r_90_percentile;
};

struct ShowerComparison
{
    // Penetration depth comparison
    double z_10_rel_diff;
    double z_50_rel_diff;
    double z_90_rel_diff;
    double z_rms_rel_diff;

    // Radial spread comparison
    double r_rms_rel_diff;
    double r_10_rel_diff;
    double r_50_rel_diff;
    double r_90_rel_diff;
};

//---------------------------------------------------------------------------//
// Calculate shower profile metrics from 2D r-z histogram
//---------------------------------------------------------------------------//
ShowerMetrics calculate_shower_metrics_rz(TH2D* h)
{
    ShowerMetrics metrics;

    // Get z and r projections for percentile calculations
    TH1D* proj_z = h->ProjectionX("_pz_temp");
    TH1D* proj_r = h->ProjectionY("_pr_temp");

    // Calculate z percentiles (penetration depth)
    double quantiles_z[3];
    double probSum[3] = {0.1, 0.5, 0.9};
    proj_z->GetQuantiles(3, quantiles_z, probSum);
    metrics.z_10_percentile = quantiles_z[0];
    metrics.z_50_percentile = quantiles_z[1];
    metrics.z_90_percentile = quantiles_z[2];
    metrics.z_mean = proj_z->GetMean();
    metrics.z_rms = proj_z->GetRMS();

    // Calculate r percentiles (radial spread)
    double quantiles_r[3];
    proj_r->GetQuantiles(3, quantiles_r, probSum);
    metrics.r_10_percentile = quantiles_r[0];
    metrics.r_50_percentile = quantiles_r[1];
    metrics.r_90_percentile = quantiles_r[2];
    metrics.r_mean = proj_r->GetMean();
    metrics.r_rms = proj_r->GetRMS();

    delete proj_z;
    delete proj_r;

    return metrics;
}

//---------------------------------------------------------------------------//
// Calculate shower profile metrics from 2D phi-z histogram
//---------------------------------------------------------------------------//
ShowerMetrics calculate_shower_metrics_phiz(TH2D* h)
{
    ShowerMetrics metrics;

    // Get z and phi projections
    TH1D* proj_z = h->ProjectionX("_pz_temp");
    TH1D* proj_phi = h->ProjectionY("_pphi_temp");

    // Calculate z percentiles (penetration depth)
    double quantiles_z[3];
    double probSum[3] = {0.1, 0.5, 0.9};
    proj_z->GetQuantiles(3, quantiles_z, probSum);
    metrics.z_10_percentile = quantiles_z[0];
    metrics.z_50_percentile = quantiles_z[1];
    metrics.z_90_percentile = quantiles_z[2];
    metrics.z_mean = proj_z->GetMean();
    metrics.z_rms = proj_z->GetRMS();

    // For phi distribution, we care about uniformity (RMS)
    // Phi mean is not physically meaningful (depends on coordinate system)
    // But phi RMS tells us about azimuthal spread
    metrics.r_10_percentile = 0;  // Not applicable for phi-z
    metrics.r_50_percentile = 0;
    metrics.r_90_percentile = 0;
    metrics.r_mean = proj_phi->GetMean();  // Store phi mean for comparison
    metrics.r_rms = proj_phi->GetRMS();  // Azimuthal spread

    delete proj_z;
    delete proj_phi;

    return metrics;
}

//---------------------------------------------------------------------------//
// Compare shower metrics between two histograms
//---------------------------------------------------------------------------//
ShowerComparison compare_shower_metrics(ShowerMetrics const& m1,
                                        ShowerMetrics const& m2,
                                        TH2D*,
                                        TH2D*)
{
    ShowerComparison comp;

    // Penetration depth comparisons (relative differences for percentiles)
    comp.z_10_rel_diff = (m1.z_10_percentile != 0)
                             ? (m1.z_10_percentile - m2.z_10_percentile)
                                   / m1.z_10_percentile * 100
                             : 0;
    comp.z_50_rel_diff = (m1.z_50_percentile != 0)
                             ? (m1.z_50_percentile - m2.z_50_percentile)
                                   / m1.z_50_percentile * 100
                             : 0;
    comp.z_90_rel_diff = (m1.z_90_percentile != 0)
                             ? (m1.z_90_percentile - m2.z_90_percentile)
                                   / m1.z_90_percentile * 100
                             : 0;

    comp.z_rms_rel_diff
        = (m1.z_rms != 0) ? (m1.z_rms - m2.z_rms) / m1.z_rms * 100 : 0;

    // Radial spread comparisons
    comp.r_rms_rel_diff
        = (m1.r_rms != 0) ? (m1.r_rms - m2.r_rms) / m1.r_rms * 100 : 0;

    comp.r_10_rel_diff = (m1.r_10_percentile != 0)
                             ? (m1.r_10_percentile - m2.r_10_percentile)
                                   / m1.r_10_percentile * 100
                             : 0;
    comp.r_50_rel_diff = (m1.r_50_percentile != 0)
                             ? (m1.r_50_percentile - m2.r_50_percentile)
                                   / m1.r_50_percentile * 100
                             : 0;
    comp.r_90_rel_diff = (m1.r_90_percentile != 0)
                             ? (m1.r_90_percentile - m2.r_90_percentile)
                                   / m1.r_90_percentile * 100
                             : 0;

    return comp;
}

//---------------------------------------------------------------------------//
// Print shower comparison metrics
//---------------------------------------------------------------------------//
bool print_shower_metrics(TString plot_type,
                          ShowerMetrics const& m1,
                          ShowerMetrics const&,
                          ShowerComparison const& comp)
{
    bool all_passed = true;
    double threshold = 3.0;  // 3% threshold for all tests

    // Only print z metrics for r-z plots
    if (m1.z_mean != 0)
    {
        // Hard checks on penetration depth
        bool z_10_pass = TMath::Abs(comp.z_10_rel_diff) < threshold;
        bool z_50_pass = TMath::Abs(comp.z_50_rel_diff) < threshold;
        bool z_90_pass = TMath::Abs(comp.z_90_rel_diff) < threshold;

        std::cout << "  [" << (z_10_pass ? "PASS" : "FAIL") << "] "
                  << plot_type << " z 10th percentile: |" << comp.z_10_rel_diff
                  << "%| (threshold: <3%)" << std::endl;
        std::cout << "  [" << (z_50_pass ? "PASS" : "FAIL") << "] "
                  << plot_type << " z 50th percentile: |" << comp.z_50_rel_diff
                  << "%| (threshold: <3%)" << std::endl;
        std::cout << "  [" << (z_90_pass ? "PASS" : "FAIL") << "] "
                  << plot_type << " z 90th percentile: |" << comp.z_90_rel_diff
                  << "%| (threshold: <3%)" << std::endl;

        all_passed = all_passed && z_10_pass && z_50_pass && z_90_pass;
    }

    // For phi-z plots, skip radial tests (no validation, just informational)
    if (!plot_type.Contains("phiz"))
    {
        // For r-z and x-y plots, display radial spread
        if (m1.r_10_percentile != 0)  // Only print percentiles if available
        {
            // Hard checks on radial spread
            bool r_10_pass = TMath::Abs(comp.r_10_rel_diff) < threshold;
            bool r_50_pass = TMath::Abs(comp.r_50_rel_diff) < threshold;
            bool r_90_pass = TMath::Abs(comp.r_90_rel_diff) < threshold;

            std::cout << "  [" << (r_10_pass ? "PASS" : "FAIL") << "] "
                      << plot_type << " r 10th percentile: |"
                      << comp.r_10_rel_diff << "%| (threshold: <3%)"
                      << std::endl;
            std::cout << "  [" << (r_50_pass ? "PASS" : "FAIL") << "] "
                      << plot_type << " r 50th percentile: |"
                      << comp.r_50_rel_diff << "%| (threshold: <3%)"
                      << std::endl;
            std::cout << "  [" << (r_90_pass ? "PASS" : "FAIL") << "] "
                      << plot_type << " r 90th percentile: |"
                      << comp.r_90_rel_diff << "%| (threshold: <3%)"
                      << std::endl;

            all_passed = all_passed && r_10_pass && r_50_pass && r_90_pass;
        }
    }

    return all_passed;
}

//---------------------------------------------------------------------------//
// 1D histogram data processing
//---------------------------------------------------------------------------//
double loop(TString file, TH1D* hist, TString plot_type)
{
    auto tfile = TFile::Open(file.Data(), "read");
    if (!tfile || tfile->IsZombie())
    {
        std::cerr << "Error: Cannot open file " << file.Data() << std::endl;
        return 0.0;
    }

    // Create TTreeReader
    TTreeReader reader("EVENT", tfile);

    // Define TTreeReaderValues for different branches
    TTreeReaderValue<std::vector<dd4hep::sim::Geant4Particle*>> mcParticles(
        reader, "MCParticles");
    TTreeReaderValue<std::vector<dd4hep::sim::Geant4Calorimeter::Hit*>>
        ecalEndcapHits(reader, "EcalEndcapHits");

    // For calorimeter deposited energy sum validation
    double total_deposited_energy = 0;

    // Event loop
    while (reader.Next())
    {
        if (plot_type == "pt" || plot_type == "eta" || plot_type == "phi")
        {
            // Analyze MC particles (primaries only)
            for (auto const& particle : *mcParticles)
            {
                if (particle)
                {
                    // Filter for primary particles only (no parents)
                    if (particle->parents.size() != 0)
                    {
                        continue;
                    }

                    // Get momentum components
                    double px = particle->psx;
                    double py = particle->psy;
                    double pz = particle->psz;

                    if (plot_type == "pt")
                    {
                        double pt = TMath::Sqrt(px * px + py * py);
                        hist->Fill(pt);
                    }
                    else if (plot_type == "eta")
                    {
                        double p = TMath::Sqrt(px * px + py * py + pz * pz);
                        if (p > 0 && TMath::Abs(pz) < p)
                        {
                            double theta = TMath::ACos(pz / p);
                            if (theta > 0 && theta < TMath::Pi())
                            {
                                double eta
                                    = -TMath::Log(TMath::Tan(0.5 * theta));
                                hist->Fill(eta);
                            }
                        }
                    }
                    else if (plot_type == "phi")
                    {
                        double phi = TMath::ATan2(py, px);
                        hist->Fill(phi);
                    }
                }
            }
        }
        else if (plot_type == "ecal_endcap_energy")
        {
            double event_deposited_energy = 0;
            for (auto const& hit : *ecalEndcapHits)
                if (hit)
                    event_deposited_energy += hit->energyDeposit;

            hist->Fill(event_deposited_energy);
            total_deposited_energy += event_deposited_energy;
        }
    }

    // Return total deposited energy for validation (no verbose output)

    tfile->Close();
    return total_deposited_energy;
}

//---------------------------------------------------------------------------//
/*!
 * Compare 1D histograms between two ROOT files.
 *
 * Supported plot types: pt, eta, phi, ecal_endcap_energy
 */
bool compare_1D_histos(TString config1_rootfile,
                       TString config2_rootfile,
                       TString plot_type)
{
    // Prepare labels - extract basename and remove .root extension
    TString config1_label = gSystem->BaseName(config1_rootfile);
    config1_label.ReplaceAll(".root", "");
    TString config2_label = gSystem->BaseName(config2_rootfile);
    config2_label.ReplaceAll(".root", "");

    // Configure histogram parameters based on plot type
    double hist_bin_min, hist_bin_max;
    int hist_n_bins;
    TString hist_x_axis, hist_plot_title;
    TH1D *h_config1, *h_config2;

    if (plot_type == "pt")
    {
        hist_bin_min = 4000;
        hist_bin_max = 5000;
        hist_n_bins = 50;
        hist_x_axis = "p_{T} [MeV]";
        hist_plot_title = "MC Particle p_{T} Distribution (DD4hep Simulation)";
    }
    else if (plot_type == "eta")
    {
        hist_bin_min = 1.8;
        hist_bin_max = 2.3;
        hist_n_bins = 25;
        hist_x_axis = "#eta";
        hist_plot_title = "MC Particle #eta Distribution (DD4hep Simulation)";
    }
    else if (plot_type == "phi")
    {
        hist_bin_min = -TMath::Pi();
        hist_bin_max = TMath::Pi();
        hist_n_bins = n_bins;
        hist_x_axis = "#phi";
        hist_plot_title = "MC Particle #phi Distribution (DD4hep Simulation)";
    }
    else if (plot_type == "ecal_endcap_energy")
    {
        hist_bin_min = 200;
        hist_bin_max = 400;
        hist_n_bins = n_bins;
        hist_x_axis = "Energy [MeV]";
        hist_plot_title
            = "ECAL Endcap Total Energy per Event (DD4hep Simulation)";
    }
    else
    {
        std::cerr << "Error: Unknown plot type " << plot_type << std::endl;
        return false;
    }

    // Create histograms
    h_config1
        = new TH1D("Config1", "", hist_n_bins, hist_bin_min, hist_bin_max);
    h_config2
        = new TH1D("Config2", "", hist_n_bins, hist_bin_min, hist_bin_max);

    // Process data
    double total_energy1 = loop(config1_rootfile, h_config1, plot_type);
    double total_energy2 = loop(config2_rootfile, h_config2, plot_type);

    // Calculate and print statistical metrics
    ComparisonMetrics metrics = calculate_1D_metrics(h_config1, h_config2);
    bool passed = print_1D_metrics(plot_type, metrics);

    // Additional validation for energy
    if (plot_type == "ecal_endcap_energy")
    {
        // Hard check on total energy difference < 3%
        double rel_diff = 0.0;
        if (total_energy1 != 0.0)
        {
            rel_diff
                = TMath::Abs((total_energy1 - total_energy2) / total_energy1)
                  * 100.0;
        }
        passed = rel_diff < 3.0;
        std::cout << "  [" << (passed ? "PASS" : "FAIL") << "] "
                  << "Total energy: |rel diff| = " << rel_diff
                  << "% (threshold: <3%)" << std::endl;
    }

    // Create relative error histograms
    auto h_config1_rel_err = new TH1D(
        "Config1 rel. err.", "", hist_n_bins, hist_bin_min, hist_bin_max);
    auto h_config1_rel_err_3s = new TH1D(
        "Config1 rel. err. 3sigma", "", hist_n_bins, hist_bin_min, hist_bin_max);

    for (int i = 0; i < hist_n_bins; i++)
    {
        double error = h_config1->GetBinError(i);
        double value = h_config1->GetBinContent(i);
        double rel_err = value ? error / value : 0;

        h_config1_rel_err->SetBinContent(i, 0);
        h_config1_rel_err->SetBinError(i, rel_err * 100);
        h_config1_rel_err_3s->SetBinContent(i, 0);
        h_config1_rel_err_3s->SetBinError(i, 3 * rel_err * 100);
    }

    // Create relative difference histogram [(Config1 - Config2) / Config1]
    auto h_rel_diff = (TH1D*)h_config1->Clone();
    h_rel_diff->Add(h_config2, -1);
    h_rel_diff->Divide(h_config1);
    h_rel_diff->Scale(100);  // In [%]

    // Create canvas
    auto canvas = new TCanvas("c1", "c1", 750, 600);
    canvas->Divide(1, 2);

    // Create top pad and move to it
    auto pad_top = new TPad("pad1", "", 0.0, 0.3, 1.0, 1.0);
    pad_top->SetBottomMargin(0.02);
    pad_top->SetLeftMargin(0.11);
    pad_top->Draw();
    pad_top->cd();

    // Histograms attributes
    auto const config2_color = kAzure + 1;
    h_config2->SetLineColor(config2_color);
    h_config2->SetLineWidth(2);
    h_config1->SetMarkerStyle(46);
    h_config1->SetMarkerSize(1.6);

    h_config1->GetXaxis()->SetLabelOffset(99);
    h_config1->GetYaxis()->SetLabelOffset(0.007);
    h_config1->GetYaxis()->CenterTitle();

    // Disable default ROOT stats box
    h_config1->SetStats(0);
    h_config2->SetStats(0);

    // Draw histograms
    h_config1->Draw("PE2");
    h_config2->Draw("hist sames");

    auto legend_top = new TLegend(0.55, 0.60, 0.88, 0.88);
    legend_top->SetHeader("Statistics", "C");
    legend_top->SetTextSize(0.035);

    // Add entry for config1
    TString stats1 = Form("%s: Mean=%.1f, RMS=%.1f",
                          config1_label.Data(),
                          h_config1->GetMean(),
                          h_config1->GetRMS());
    legend_top->AddEntry(h_config1, stats1, "p");

    // Add entry for config2
    TString stats2 = Form("%s: Mean=%.1f, RMS=%.1f",
                          config2_label.Data(),
                          h_config2->GetMean(),
                          h_config2->GetRMS());
    legend_top->AddEntry(h_config2, stats2, "l");

    legend_top->SetMargin(0.12);
    legend_top->SetBorderSize(1);
    legend_top->SetFillStyle(1001);
    legend_top->SetFillColorAlpha(kWhite, 0.9);
    legend_top->Draw();

    auto title_text = new TText(0.17, 0.92, hist_plot_title);
    title_text->SetNDC();
    title_text->SetTextColor(kGray);
    title_text->Draw();

    auto commit_text = new TLatex(0.67, 0.92, commit_hash);
    commit_text->SetNDC();
    commit_text->SetTextColor(kGray);
    commit_text->Draw();

    // Redraw axis above the histogram lines
    pad_top->RedrawAxis();
    // Move back to canvas
    canvas->cd();

    // Create bottom pad and move to it
    auto pad_bottom = new TPad("pad2", "", 0.0, 0.0, 1.0, 0.3);
    pad_bottom->SetTopMargin(0.02);
    pad_bottom->SetBottomMargin(0.33);
    pad_bottom->SetLeftMargin(0.11);
    pad_bottom->Draw();
    pad_bottom->cd();

    h_config1_rel_err_3s->GetXaxis()->SetTitle(hist_x_axis);
    h_config1_rel_err_3s->GetXaxis()->CenterTitle();
    h_config1_rel_err_3s->GetXaxis()->SetTitleSize(0.14);
    h_config1_rel_err_3s->GetXaxis()->SetTitleOffset(1.1);
    h_config1_rel_err_3s->GetXaxis()->SetLabelSize(0.1153);
    h_config1_rel_err_3s->GetXaxis()->SetLabelOffset(0.02);
    h_config1_rel_err_3s->GetXaxis()->SetTickLength(0.07);

    h_config1_rel_err_3s->GetYaxis()->SetTitle("Rel. Diff. (%)");
    h_config1_rel_err_3s->GetYaxis()->CenterTitle();
    h_config1_rel_err_3s->GetYaxis()->SetTitleSize(0.131);
    h_config1_rel_err_3s->GetYaxis()->SetTitleOffset(0.415);
    h_config1_rel_err_3s->GetYaxis()->SetLabelSize(0.116);
    h_config1_rel_err_3s->GetYaxis()->SetLabelOffset(0.008);
    h_config1_rel_err_3s->GetYaxis()->SetTickLength(0.04);
    h_config1_rel_err_3s->GetYaxis()->SetNdivisions(503);
    h_config1_rel_err_3s->GetYaxis()->SetRangeUser(-5, 5);

    h_config1_rel_err_3s->SetLineColorAlpha(kGray, 0.7);
    h_config1_rel_err_3s->SetFillColorAlpha(kGray, 0.7);
    h_config1_rel_err_3s->SetMarkerSize(0);
    h_config1_rel_err->SetLineColorAlpha(kGray + 1, 0.7);
    h_config1_rel_err->SetFillColorAlpha(kGray + 1, 0.7);
    h_config1_rel_err->SetMarkerSize(0);
    h_config1_rel_err->GetYaxis()->SetRangeUser(-5, 5);
    h_rel_diff->SetLineColor(config2_color);

    // Draw stat. err. and rel. diff. histograms
    h_config1_rel_err_3s->Draw("hist E2");
    h_config1_rel_err->Draw("hist E2 sames");
    h_rel_diff->Draw("hist sames");

    pad_bottom->RedrawAxis();
    canvas->Print(config1_label + "_" + config2_label + "_" + plot_type
                  + ".png");

    // Clean up
    delete canvas;
    delete h_config1;
    delete h_config2;
    delete h_config1_rel_err;
    delete h_config1_rel_err_3s;
    delete h_rel_diff;

    return passed;
}

//---------------------------------------------------------------------------//
// 2D histogram helper functions
//---------------------------------------------------------------------------//
void SetAxisStyle(TAxis* axis,
                  double titleSize,
                  double titleOffset,
                  double labelSize,
                  double labelOffset,
                  double tickLength,
                  int nDivisions,
                  bool isXaxis)
{
    if (isXaxis)
        axis->SetMaxDigits(4);
    axis->SetTitleSize(titleSize);
    axis->SetTitleOffset(titleOffset);
    axis->SetLabelSize(labelSize);
    axis->SetLabelOffset(labelOffset);
    axis->SetTickLength(tickLength);
    axis->SetNdivisions(nDivisions);
}

void DrawEntryNote(TCanvas* c, TH2D*, int pad_num, TString primary)
{
    TLatex* text = new TLatex(0.37, 0.88, Form("%s", primary.Data()));
    text->SetTextSize(0.04);
    text->SetNDC();
    c->cd(pad_num);
    text->Draw();
}

// Helper functions to fill 2D histograms from hit collections
template<typename HitType>
void fill_2D_rz_hist_impl(TH2D* h, std::vector<HitType*> const& hits)
{
    for (auto const& hit : hits)
    {
        if (hit)
        {
            double x = hit->position.x();
            double y = hit->position.y();
            double z = hit->position.z();
            double r = TMath::Sqrt(x * x + y * y);
            h->Fill(z, r);
        }
    }
}

void fill_2D_rz_hist(
    TH2D* h,
    TTreeReaderValue<std::vector<dd4hep::sim::Geant4Calorimeter::Hit*>>& hits)
{
    fill_2D_rz_hist_impl(h, *hits);
}

void fill_2D_rz_tracker_hist(
    TH2D* h,
    TTreeReaderValue<std::vector<dd4hep::sim::Geant4Tracker::Hit*>>& hits)
{
    fill_2D_rz_hist_impl(h, *hits);
}

void fill_2D_phiz_hist(
    TH2D* h,
    TTreeReaderValue<std::vector<dd4hep::sim::Geant4Calorimeter::Hit*>>& hits)
{
    for (auto const& hit : *hits)
    {
        if (hit)
        {
            double x = hit->position.x();
            double y = hit->position.y();
            double z = hit->position.z();
            double phi = TMath::ATan2(y, x);
            h->Fill(z, phi);
        }
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //
// Create 2D histograms from DD4hep data
void create_2D_histograms(TString file, TH2D* h, TString plot_type)
{
    auto tfile = TFile::Open(file.Data(), "read");
    if (!tfile || tfile->IsZombie())
    {
        std::cerr << "Error: Cannot open file " << file.Data() << std::endl;
        return;
    }

    // Create TTreeReader
    TTreeReader reader("EVENT", tfile);

    // All tracker collections
    TTreeReaderValue<std::vector<dd4hep::sim::Geant4Tracker::Hit*>>
        siVertexBarrelHits(reader, "SiVertexBarrelHits");
    TTreeReaderValue<std::vector<dd4hep::sim::Geant4Tracker::Hit*>>
        siVertexEndcapHits(reader, "SiVertexEndcapHits");
    TTreeReaderValue<std::vector<dd4hep::sim::Geant4Tracker::Hit*>>
        siTrackerBarrelHits(reader, "SiTrackerBarrelHits");
    TTreeReaderValue<std::vector<dd4hep::sim::Geant4Tracker::Hit*>>
        siTrackerEndcapHits(reader, "SiTrackerEndcapHits");
    TTreeReaderValue<std::vector<dd4hep::sim::Geant4Tracker::Hit*>>
        siTrackerForwardHits(reader, "SiTrackerForwardHits");

    // Calorimeter collection
    TTreeReaderValue<std::vector<dd4hep::sim::Geant4Calorimeter::Hit*>>
        ecalEndcapHits(reader, "EcalEndcapHits");

    // Event loop
    while (reader.Next())
    {
        // Fill tracker r-z histogram from all tracker collections
        if (plot_type == "tracker_rz")
        {
            fill_2D_rz_tracker_hist(h, siVertexBarrelHits);
            fill_2D_rz_tracker_hist(h, siVertexEndcapHits);
            fill_2D_rz_tracker_hist(h, siTrackerBarrelHits);
            fill_2D_rz_tracker_hist(h, siTrackerEndcapHits);
            fill_2D_rz_tracker_hist(h, siTrackerForwardHits);
        }

        if (plot_type == "ecal_endcap_rz")
            fill_2D_rz_hist(h, ecalEndcapHits);
        else if (plot_type == "ecal_endcap_phiz")
            fill_2D_phiz_hist(h, ecalEndcapHits);
    }

    tfile->Close();
}

//---------------------------------------------------------------------------//
/*!
 * Compare 2D spatial histograms between two ROOT files.
 *
 * Supported plot types: tracker_rz, ecal_endcap_rz, ecal_endcap_phiz
 */
bool compare_2D_histos(TString f1_name,
                       TString f2_name,
                       TString plot_type,
                       TString primary)
{
    // Extract basenames for output labels
    TString h1_label = gSystem->BaseName(f1_name);
    h1_label.ReplaceAll(".root", "");
    TString h2_label = gSystem->BaseName(f2_name);
    h2_label.ReplaceAll(".root", "");

    // Configure histogram parameters based on plot type
    TString hist_name1, hist_name2, hist_title, hist_axes;
    int nbins_x, nbins_y;
    double xmin, xmax, ymin, ymax;
    TH2D *h1, *h2;

    if (plot_type == "tracker_rz")
    {
        hist_name1 = "h1_tracker_rz";
        hist_name2 = "h2_tracker_rz";
        hist_title = "Tracker Hit r-z Distribution;z [mm];r [mm]";
        nbins_x = 40;
        xmin = -1500;
        xmax = 2500;
        nbins_y = 30;
        ymin = 0;
        ymax = 1500;
    }
    else if (plot_type == "ecal_endcap_rz")
    {
        hist_name1 = "h1_ecal_endcap_rz";
        hist_name2 = "h2_ecal_endcap_rz";
        hist_title = "ECAL Endcap Hit r-z Distribution;z [mm];r [mm]";
        nbins_x = 30;
        xmin = 1400;
        xmax = 2000;
        nbins_y = 30;
        ymin = 200;  // Lower range to capture full shower at eta 2.0-2.1
        ymax = 800;
    }
    else if (plot_type == "ecal_endcap_phiz")
    {
        hist_name1 = "h1_ecal_endcap_phiz";
        hist_name2 = "h2_ecal_endcap_phiz";
        hist_title = "ECAL Endcap Hit #phi-z Distribution;z [mm];#phi [rad]";
        nbins_x = 30;
        xmin = 1400;
        xmax = 2000;
        nbins_y = 24;
        ymin = -TMath::Pi();
        ymax = TMath::Pi();
    }
    else
    {
        std::cerr << "Error: Unknown plot type " << plot_type << std::endl;
        return false;
    }

    // Create histograms with configured parameters
    h1 = new TH2D(
        hist_name1, hist_title, nbins_x, xmin, xmax, nbins_y, ymin, ymax);
    h2 = new TH2D(
        hist_name2, hist_title, nbins_x, xmin, xmax, nbins_y, ymin, ymax);

    // Fill histograms from DD4hep data
    create_2D_histograms(f1_name, h1, plot_type);
    create_2D_histograms(f2_name, h2, plot_type);

    // Calculate and print shower profile metrics (calorimeters only)
    bool passed = true;  // Default to pass for non-validated plots (e.g.,
                         // tracker)
    if (plot_type.Contains("ecal"))
    {
        ShowerMetrics shower1, shower2;
        if (plot_type.Contains("_rz"))
        {
            shower1 = calculate_shower_metrics_rz(h1);
            shower2 = calculate_shower_metrics_rz(h2);
        }
        else if (plot_type.Contains("_phiz"))
        {
            shower1 = calculate_shower_metrics_phiz(h1);
            shower2 = calculate_shower_metrics_phiz(h2);
        }

        ShowerComparison shower_comp
            = compare_shower_metrics(shower1, shower2, h1, h2);
        passed = print_shower_metrics(plot_type, shower1, shower2, shower_comp);
    }

    // Initialize canvas
    TCanvas* c = new TCanvas("c", "c", 1200, 600);
    c->Divide(2, 1);

    // Turn on Stat Box and position it
    gStyle->SetOptStat(1);
    gStyle->SetStatX(0.85);
    gStyle->SetStatY(0.85);

    // Switch to pad 1 and draw histogram from first file
    c->cd(1);
    h1->SetTitle(h1_label);
    h1->Draw("colz");

    // Set z-axis range and log scale
    h1->SetMaximum(1e5);
    h1->SetMinimum(1);
    gPad->SetLogz();

    // Set margins on pad and style the axes
    gPad->SetMargin(0.13, 0.13, 0.13, 0.13);
    SetAxisStyle(
        h1->GetXaxis(), 0.06, 0.415, 0.04, 0.02, 0.04, 3015, true);  // X axis
    SetAxisStyle(
        h1->GetYaxis(), 0.06, 0.415, 0.04, 0.02, 0.04, 1006, false);  // Y axis

    // Draw additional information label below histogram title
    DrawEntryNote(c, h1, 1, primary);

    // Switch to pad 2 and draw histogram from second file
    c->cd(2);
    h2->SetTitle(h2_label);
    h2->Draw("colz");

    // Set z-axis range and log scale
    h2->SetMaximum(1e5);
    h2->SetMinimum(1);
    gPad->SetLogz();

    // Set margins on pad and style the axes
    gPad->SetMargin(0.13, 0.13, 0.13, 0.13);
    SetAxisStyle(
        h2->GetXaxis(), 0.04, 1.6, 0.04, 0.02, 0.04, 3015, true);  // X axis
    SetAxisStyle(
        h2->GetYaxis(), 0.04, 1.6, 0.04, 0.02, 0.04, 1006, false);  // Y axis

    // Draw additional information label below histogram title
    DrawEntryNote(c, h2, 2, primary);

    // Save image with appropriate name
    c->Print(h1_label + "_" + h2_label + "_" + plot_type + ".png");

    // Clean up
    delete c;
    delete h1;
    delete h2;

    return passed;
}

//---------------------------------------------------------------------------//
/*!
 * Main entry point: Compare all relevant histograms between two
 * configurations.
 *
 * Validation tests performed (6 total):
 * - Initial distributions (pt, eta, phi): KS test (p-value > 0.05)
 * - Calorimeter energy (ecal_endcap_energy): Total energy difference < 3%
 * - Shower profiles (ecal_endcap_rz, ecal_endcap_phiz): Percentiles within 3%
 * - Tracker distribution (tracker_rz): Informational only
 */
void compare_benchmarks(TString config1_rootfile, TString config2_rootfile)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "   DD4hep Validation Benchmark" << std::endl;
    std::cout << "========================================\n" << std::endl;

    bool all_tests_passed = true;
    int num_tests = 0;
    int num_passed = 0;

    // Test 1-3: Initial kinematic distributions (KS test)
    std::cout << "[1/3] Initial Particle Distributions (KS Test)" << std::endl;
    bool test_pt = compare_1D_histos(config1_rootfile, config2_rootfile, "pt");
    num_tests++;
    if (test_pt)
        num_passed++;
    all_tests_passed = all_tests_passed && test_pt;

    bool test_eta
        = compare_1D_histos(config1_rootfile, config2_rootfile, "eta");
    num_tests++;
    if (test_eta)
        num_passed++;
    all_tests_passed = all_tests_passed && test_eta;

    bool test_phi
        = compare_1D_histos(config1_rootfile, config2_rootfile, "phi");
    num_tests++;
    if (test_phi)
        num_passed++;
    all_tests_passed = all_tests_passed && test_phi;

    // Test 4: ECAL endcap energy distribution (total energy difference < 3%)
    std::cout << "\n[2/3] Calorimeter Energy Deposition" << std::endl;
    bool test_energy = compare_1D_histos(
        config1_rootfile, config2_rootfile, "ecal_endcap_energy");
    num_tests++;
    if (test_energy)
        num_passed++;
    all_tests_passed = all_tests_passed && test_energy;

    // Test 5-6: ECAL endcap shower profiles (2D histograms)
    std::cout << "\n[3/3] Shower Profile Distributions" << std::endl;
    bool test_shower_rz = compare_2D_histos(config1_rootfile,
                                            config2_rootfile,
                                            "ecal_endcap_rz",
                                            "DD4hep ECAL Endcap Analysis");
    num_tests++;
    if (test_shower_rz)
        num_passed++;
    all_tests_passed = all_tests_passed && test_shower_rz;

    bool test_shower_phiz = compare_2D_histos(config1_rootfile,
                                              config2_rootfile,
                                              "ecal_endcap_phiz",
                                              "DD4hep ECAL Endcap Analysis");
    num_tests++;
    if (test_shower_phiz)
        num_passed++;
    all_tests_passed = all_tests_passed && test_shower_phiz;

    // Print summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "   Validation Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Tests passed: " << num_passed << "/" << num_tests
              << std::endl;
    std::cout << "Overall: " << (all_tests_passed ? "PASS ✓" : "FAIL ✗")
              << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Return exit code for CI/CD
    if (!all_tests_passed)
    {
        exit(1);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Main entry point for compiled executable.
 */
int main(int argc, char** argv)
{
    TString file1, file2;

    // Try to read from command-line arguments first
    if (argc == 3)
    {
        file1 = argv[1];
        file2 = argv[2];
    }
    // Fall back to environment variables
    else if (argc == 1)
    {
        char const* env_file1 = std::getenv("DDCELER_FILE1");
        char const* env_file2 = std::getenv("DDCELER_FILE2");

        if (!env_file1 || !env_file2)
        {
            std::cerr << "Usage: " << argv[0] << " <file1.root> <file2.root>"
                      << std::endl;
            std::cerr << "\nOr set environment variables DDCELER_FILE1 and "
                         "DDCELER_FILE2"
                      << std::endl;
            std::cerr
                << "\nCompares two DD4hep simulation outputs for validation."
                << std::endl;
            std::cerr << "Example: " << argv[0]
                      << " output_celeritas.root output_geant4.root"
                      << std::endl;
            std::cerr << "Example: DDCELER_FILE1=file1.root "
                         "DDCELER_FILE2=file2.root "
                      << argv[0] << std::endl;
            return 1;
        }

        file1 = env_file1;
        file2 = env_file2;
    }
    else
    {
        std::cerr << "Usage: " << argv[0] << " <file1.root> <file2.root>"
                  << std::endl;
        std::cerr
            << "\nOr set environment variables DDCELER_FILE1 and DDCELER_FILE2"
            << std::endl;
        std::cerr << "\nCompares two DD4hep simulation outputs for validation."
                  << std::endl;
        std::cerr << "Example: " << argv[0]
                  << " output_celeritas.root output_geant4.root" << std::endl;
        std::cerr
            << "Example: DDCELER_FILE1=file1.root DDCELER_FILE2=file2.root "
            << argv[0] << std::endl;
        return 1;
    }

    // Check if files exist
    if (gSystem->AccessPathName(file1))
    {
        std::cerr << "Error: Cannot access file: " << file1 << std::endl;
        return 1;
    }
    if (gSystem->AccessPathName(file2))
    {
        std::cerr << "Error: Cannot access file: " << file2 << std::endl;
        return 1;
    }

    // Load DD4hep dictionaries for ROOT to properly access DD4hep classes
    gSystem->Load("libDDG4Plugins.so");
    gSystem->Load("libDDG4.so");

    // Run comparison
    compare_benchmarks(file1, file2);

    return 0;
}
