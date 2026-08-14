import unittest

from bedrock_rl.sdg.policy.recipes import (
    choose_recipe, ingredient_cells, item_stack_size, minecraft_data,
    recipe_variants,
)
from bedrock_rl.env.catalog import resolve_blocks, resolve_items


class RecipeRegistryTests(unittest.TestCase):
    def test_complete_standard_registry_is_pinned(self):
        data = minecraft_data()
        self.assertEqual(data["minecraft"], "1.11.2")
        self.assertEqual(
            data["source"]["commit"],
            "79f6aab42037d09acd09529baed87b62ccd48908")
        self.assertEqual(len(data["recipes"]), 360)
        self.assertEqual(len({row["output"][0]
                              for row in data["recipes"]}), 210)

    def test_every_layout_fits_a_vanilla_crafting_grid(self):
        for recipe in minecraft_data()["recipes"]:
            if recipe["shaped"]:
                self.assertLessEqual(len(recipe["shape"]), 3)
                self.assertLessEqual(len(recipe["shape"][0]), 3)
            else:
                self.assertLessEqual(len(recipe["ingredients"]), 9)

    def test_gui_expert_selects_any_available_recipe(self):
        recipe = choose_recipe(
            "diamond_pickaxe",
            [264, 280, 0, 0, 0, 0, 0, 0, 0],
            [3, 2, 0, 0, 0, 0, 0, 0, 0],
            3,
        )
        self.assertIsNotNone(recipe)
        self.assertEqual(ingredient_cells(recipe, 3), (
            (264, 32767, (0, 1, 2)), (280, 32767, (4, 7))))

    def test_metadata_variants_use_the_matching_hotbar_stack(self):
        recipe = choose_recipe(
            "planks",
            [17, 17, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0],
            2,
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            output_meta=0,
        )
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe["shape"], [[[17, 0]]])
        self.assertEqual(ingredient_cells(recipe, 2), ((17, 0, (0,)),))

    def test_generic_wood_recipes_accept_every_plank_species(self):
        for output in ("stick", "crafting_table", "chest",
                       "wooden_pickaxe"):
            recipe = recipe_variants(output)[0]
            planks = [value for row in recipe.get("shape", ())
                      for value in row if value and value[0] == 5]
            self.assertTrue(planks, output)
            self.assertTrue(all(value[1] == 32767 for value in planks),
                            output)

    def test_recipe_and_name_catalogs_are_not_curated_subsets(self):
        self.assertEqual(resolve_items("golden_apple"), (322,))
        self.assertEqual(resolve_items("shield"), (442,))
        self.assertEqual(resolve_blocks("beacon"), (138,))
        self.assertTrue(recipe_variants("cake"))
        self.assertTrue(recipe_variants("comparator"))
        self.assertEqual(item_stack_size("shield"), 1)
        self.assertEqual(item_stack_size("sign"), 16)
        self.assertEqual(item_stack_size("cobblestone"), 64)

    def test_oracle_has_no_hand_maintained_recipe_dictionary(self):
        from pathlib import Path
        from bedrock_rl.sdg.policy import oracle

        source = Path(oracle.__file__).read_text()
        self.assertNotIn("GUI_RECIPES_2X2", source)
        self.assertNotIn("GUI_RECIPES_3X3", source)


if __name__ == "__main__":
    unittest.main()
