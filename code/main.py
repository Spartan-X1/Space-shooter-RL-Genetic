from custom_timer import Timer
from waves import *

class Game:
    def __init__(self):
        init_window(WINDOW_WIDTH, WINDOW_HEIGHT, 'Space shooter')
        init_audio_device()
        self.score = 0
        self.import_assets()

        # Add game timer display
        self.game_timer = Timer(inf, True, True)  # inf-second timer that repeats
        # You can adjust the duration as needed
        # First parameter: duration in seconds
        # Second parameter: repeat (True means it will restart when finished)
        # Third parameter: autostart (True means it starts immediately)

        # Add game state tracking
        self.game_over = False
        self.game_over_timer = 0

        self.assets['health'] = load_texture(join('images', 'spaceship.png'))  # You'll need a health icon

        # Add enemy ships and enemy lasers to collections
        self.lasers, self.meteors, self.explosions = [], [], []
        self.enemy_beams = []
        self.enemies, self.enemy_lasers = [], []  # New collections
        self.meteor_timer = Timer(METEOR_TIMER_DURATION, True, True, self.create_meteor)
        # Add enemy spawn timer
        self.enemy_timer = Timer(ENEMY_SPAWN_INTERVAL, True, True, self.create_enemy)  # New timer
        self.player = Player(self.assets['player'], Vector2(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2), self.shoot_laser)
        play_music_stream(self.audio['music'])
         # Initialize weapons
        self.player.weapons = [
            Weapon(self),          # Standard laser
            TripleShot(self),      # Triple shot 
            RapidFire(self)        # Rapid fire
        ]

        # Initialize wave manager
        self.wave_manager = WaveManager(self)
        self.enemy_factory = EnemyFactory(self)
        
        # Modify enemy spawn timer to be controlled by wave manager
        self.enemy_timer = Timer(ENEMY_SPAWN_INTERVAL, True, False, self.spawn_wave_enemy)
        
        # Start the first wave
        self.wave_manager.start_next_wave()

    def import_assets(self):
        self.assets = {
        'player': load_texture(join('images', 'spaceship.png')),
        'star': load_texture(join('images', 'star.png')),
        'laser': load_texture(join('images', 'laser.png')),
        'meteor': load_texture(join('images', 'meteor.png')),
        'explosion': [load_texture(join('images', 'explosion', f'{i}.png')) for i in range(1,29)],
        'font': load_font_ex(join('font', 'Stormfaze.otf'), FONT_SIZE, ffi.NULL, 0),
        
        # Add different enemy textures with meaningful names
        'enemy': load_texture(join('images','Enemies','enemyBlack1.png')),  # Default enemy
        'enemy_tank': load_texture(join('images','Enemies','enemyBlack4.png')),  # Tank enemy
        'enemy_swarm': load_texture(join('images','Enemies','enemyBlack2.png')),  # Swarm enemy  
        'enemy_sniper': load_texture(join('images','Enemies','enemyRed3.png')),  # Sniper enemy
        'enemy_bomber': load_texture(join('images','Enemies','enemyGreen5.png')),  # Bomber enemy
        
        'enemy_laser': load_texture(join('images','Lasers','laserBlue01.png')),
        'cannon_laser': load_texture(join('images','Lasers','laserBlue11.png'))
        }

        self.audio = {
            'laser': load_sound(join('audio','laser.wav')),
            'explosion': load_sound(join('audio','explosion.wav')),
            'music': load_music_stream(join('audio', 'music.wav')),
            # Optionally add enemy sound effects
            'enemy_laser': load_sound(join('audio','laser.wav'))  # Can reuse or add new sound
        }

        self.star_data = [
            (
                Vector2(randint(0, WINDOW_WIDTH),randint(0, WINDOW_HEIGHT)), # pos
                uniform(0.5, 1.6) # size
            ) for i in range(30)
        ]
    
    def draw_stars(self):
        for star in self.star_data:
            draw_texture_ex(self.assets['star'],star[0], 0, star[1], WHITE)

    def draw_timer(self):
        # Check if timer is active and has a valid start time
        if self.game_timer.active and self.game_timer.start_time > 0:
            current_time = get_time()
            
            # Calculate elapsed time instead of remaining time
            if current_time >= self.game_timer.start_time:
                elapsed_time = current_time - self.game_timer.start_time
                
                # Format elapsed time as MM:SS
                total_seconds = int(elapsed_time)
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                time_text = f"{minutes:02d}:{seconds:02d}"
                
                # Measure text size for positioning
                text_size = measure_text_ex(self.assets['font'], time_text, FONT_SIZE, 0)
                
                # Position in top right with some padding
                position = Vector2(WINDOW_WIDTH - text_size.x - 50, 20)
                
                # Draw the text
                draw_text_ex(self.assets['font'], time_text, position, FONT_SIZE, 0, WHITE)
            else:
                # Fallback display if timing is invalid
                draw_text_ex(self.assets['font'], "00:00", Vector2(WINDOW_WIDTH - 200, 20), FONT_SIZE, 0, WHITE)
        else:
            # Default display when timer is inactive
            draw_text_ex(self.assets['font'], "00:00", Vector2(WINDOW_WIDTH - 100, 20), FONT_SIZE, 0, WHITE)

    def check_weapon_unlocks(self):
        """Check if weapons should be unlocked based on wave progress"""
        current_wave = self.wave_manager.current_wave
        
        # Unlock Triple Shot at wave 2
        if current_wave >= 2 and not self.player.weapons[1].unlocked:
            self.player.weapons[1].unlocked = True
            print(f"Weapon Unlocked: {self.player.weapons[1].name}")
            
        # Unlock Rapid Fire at wave 3
        if current_wave >= 3 and not self.player.weapons[2].unlocked:
            self.player.weapons[2].unlocked = True
            print(f"Weapon Unlocked: {self.player.weapons[2].name}")
    
    def create_laser(self, pos, direction):
        """Helper method to create a laser for weapons system"""
        # Create a laser with the provided direction
        return Laser(self.assets['laser'], pos, direction)

    def shoot_laser(self, pos, direction=Vector2(0,-1)):
        # Modified to accept a direction parameter with default
        self.lasers.append(Laser(self.assets['laser'], pos, direction))
        play_sound(self.audio['laser'])

    # New method for enemy lasers
    def shoot_enemy_laser(self, texture, pos, direction, speed_multiplier=1.0, scale=1.0, damage=1):
        """Enhanced method to handle special laser properties"""
        # Create the laser with the additional parameters
        laser = EnemyLaser(texture, pos, direction, speed_multiplier, scale, damage)
        self.enemy_lasers.append(laser)
        play_sound(self.audio['enemy_laser'])
    
    def create_meteor(self):
        self.meteors.append(Meteor(self.assets['meteor']))
        
     # Modified to remove the original create_enemy method and use this one instead
    def create_enemy(self):
        """Legacy method - now just calls spawn_wave_enemy"""
        self.spawn_wave_enemy()


    # Method for the wave manager to create base enemies
    def create_base_enemy(self):
        """Create a base enemy without adding it to the game yet"""
        enemy = Enemy(self.assets['enemy'], self.shoot_enemy_laser, self.assets['enemy_laser'])
        enemy.target_player = self.player
        return enemy
    
    # New method to spawn wave-controlled enemies
    def spawn_wave_enemy(self):
        """Spawn an enemy controlled by the wave manager"""
        if self.wave_manager.wave_active:
            enemy = self.wave_manager.spawn_enemy()
            if enemy:
                self.enemies.append(enemy)

    def discard_sprites(self):
        self.lasers = [laser for laser in self.lasers if not laser.discard]
        self.meteors = [meteor for meteor in self.meteors if not meteor.discard]
        self.explosions = [explosion for explosion in self.explosions if not explosion.discard]
        # Add cleanup for enemy ships and lasers
        self.enemies = [enemy for enemy in self.enemies if not enemy.discard]
        self.enemy_lasers = [laser for laser in self.enemy_lasers if not laser.discard]

    # Add a draw_health method to visualize player health
    def draw_health(self):
        # Draw health indicators in the top-left corner
        for i in range(self.player.health):
            draw_texture_v(
                self.assets['health'],
                Vector2(20 + i * 40, 20),  # Position health icons with spacing
                WHITE
            )
    
    def draw_weapon_info(self):
        """Draw current weapon name"""
        if not self.player.weapons:
            return
            
        weapon = self.player.weapons[self.player.current_weapon_index]
        text = f"WEAPON: {weapon.name}"
        
        # Use a smaller font size
        small_font_size = FONT_SIZE // 4
        
        draw_text_ex(
            self.assets['font'],
            text,
            Vector2(20, WINDOW_HEIGHT - 40),
            small_font_size,
            0,
            WHITE
        )

    def check_beam_collision(self, beam):
        """Check if a beam hits the player and deal damage"""
        if self.player and not self.player.invulnerable:
            hit = False
            player_center = self.player.get_center()
            
            # Line-circle intersection for beam collision
            # We'll check if player is near any point along the beam
            start_to_player = Vector2(
                player_center.x - beam.start_pos.x,
                player_center.y - beam.start_pos.y
            )
            
            beam_direction = Vector2(
                beam.end_pos.x - beam.start_pos.x,
                beam.end_pos.y - beam.start_pos.y
            )
            
            beam_length = sqrt(beam_direction.x**2 + beam_direction.y**2)
            
            if beam_length > 0:
                # Normalize beam direction
                beam_direction.x /= beam_length
                beam_direction.y /= beam_length
                
                # Project player position onto beam
                projection = start_to_player.x * beam_direction.x + start_to_player.y * beam_direction.y
                
                # Clamp projection to beam length
                projection = max(0, min(projection, beam_length))
                
                # Calculate closest point on beam to player
                closest_point = Vector2(
                    beam.start_pos.x + beam_direction.x * projection,
                    beam.start_pos.y + beam_direction.y * projection
                )
                
                # Check distance from player to closest point
                distance = sqrt(
                    (player_center.x - closest_point.x)**2 + 
                    (player_center.y - closest_point.y)**2
                )
                
                # Check if player is hit
                if distance <= self.player.collision_radius:
                    hit = True
            
            if hit and self.player.take_damage():
                # Create explosion at impact point with player
                impact_pos = player_center
                self.explosions.append(ExplosionAnimation(impact_pos, self.assets['explosion']))
                play_sound(self.audio['explosion'])
                
                # Check for game over
                if self.player.health <= 0:
                    self.game_over = True

    def check_collisions(self):
        # lasers and enemies
        for laser in self.lasers:
            for enemy in self.enemies:
                if check_collision_circle_rec(enemy.get_center(), enemy.collision_radius, laser.get_rect()):
                    laser.discard = True
                    
                    # Check if enemy is destroyed
                    if enemy.take_damage():
                        enemy.discard = True
                        
                        # Create explosion
                        pos = Vector2(enemy.pos.x + enemy.size.x / 2, enemy.pos.y + enemy.size.y / 2)
                        self.explosions.append(ExplosionAnimation(pos, self.assets['explosion']))
                        play_sound(self.audio['explosion'])
                        
                        # Add to score - remaining health is a multiplier
                        # For tougher enemies that had more starting health
                        if hasattr(enemy, 'health'):
                            self.score += 100  # Base score for any enemy
                        else:
                            self.score += 50   # Fallback if health attribute is missing
                        
                        # ADD THIS LINE HERE:
                        self.wave_manager.enemy_destroyed()
        
        # lasers and meteors
        for laser in self.lasers:
            for meteor in self.meteors:
                if check_collision_circle_rec(meteor.get_center(), meteor.collision_radius, laser.get_rect()):
                    laser.discard, meteor.discard = True, True
                    pos = Vector2(laser.pos.x - laser.size.x / 2, laser.pos.y)
                    self.explosions.append(ExplosionAnimation(pos, self.assets['explosion']))
                    play_sound(self.audio['explosion'])

        # Player and enemy lasers
        for enemy_laser in self.enemy_lasers:
            if check_collision_circle_rec(self.player.get_center(), self.player.collision_radius, 
                                        enemy_laser.get_rect()):
                if self.player.take_damage():  # Only deal damage if not invulnerable
                    # Create explosion effect at the hit location
                    pos = Vector2(enemy_laser.pos.x, enemy_laser.pos.y)
                    self.explosions.append(ExplosionAnimation(pos, self.assets['explosion']))
                    play_sound(self.audio['explosion'])
                    
                    # Remove enemy laser
                    enemy_laser.discard = True
                    
                    # Check for game over
                    if self.player.health <= 0:
                        self.game_over = True

        # Player and meteors
        for meteor in self.meteors:
            if check_collision_circles(self.player.get_center(), self.player.collision_radius, 
                                    meteor.get_center(), meteor.collision_radius):
                if self.player.take_damage():  # Only deal damage if not invulnerable
                    # Create explosion effect
                    pos = Vector2(meteor.pos.x + meteor.size.x / 2, meteor.pos.y + meteor.size.y / 2)
                    self.explosions.append(ExplosionAnimation(pos, self.assets['explosion']))
                    play_sound(self.audio['explosion'])
                    
                    # Remove meteor
                    meteor.discard = True
                    
                    # Check for game over
                    if self.player.health <= 0:
                        self.game_over = True
                
        # Player and enemies
        for enemy in self.enemies:
            if check_collision_circles(self.player.get_center(), self.player.collision_radius,
                                    enemy.get_center(), enemy.collision_radius):
                if self.player.take_damage():  # Only deal damage if not invulnerable
                    # Create explosion effect
                    pos = Vector2(enemy.pos.x + enemy.size.x / 2, enemy.pos.y + enemy.size.y / 2)
                    self.explosions.append(ExplosionAnimation(pos, self.assets['explosion']))
                    play_sound(self.audio['explosion'])
                    
                    # Remove enemy
                    enemy.discard = True
                    
                    # Check for game over
                    if self.player.health <= 0:
                        self.game_over = True

    def draw_score(self):
        # Use self.score instead of get_time()
        text_size = measure_text_ex(self.assets['font'], str(self.score), FONT_SIZE, 0)
        draw_text_ex(self.assets['font'], str(self.score), Vector2(WINDOW_WIDTH / 2 - text_size.x / 2, 0), FONT_SIZE, 0, WHITE)

    def update(self):
        dt = get_frame_time()
        
        if not self.game_over:
            self.meteor_timer.update()
            self.enemy_timer.update() 
            self.game_timer.update() # Update game timer
            self.wave_manager.update(dt) #Update wave manager
            self.player.update(dt)
            self.discard_sprites()
            
            for sprite in self.lasers + self.meteors + self.enemies + self.enemy_lasers + self.explosions:
                sprite.update(dt)
            
            for beam in self.enemy_beams:
                beam.update(dt)
            self.enemy_beams = [beam for beam in self.enemy_beams if not beam.discard]
                
            self.check_collisions()
            # Add debug command to show all enemies
            # In Game.update method
            if is_key_pressed(KEY_F1):
                print("=== ACTIVE ENEMIES ===")
                for i, enemy in enumerate(self.enemies):
                    print(f"Enemy {i}: Type: {getattr(enemy, 'debug_type', 'unknown')}, "
                        f"Position: ({enemy.center_pos.x:.1f}, {enemy.center_pos.y:.1f}), "
                        f"Health: {enemy.health}, Discard: {enemy.discard}")
                print("=====================")
        else:
            # Just update explosions if there are any during game over
            for explosion in self.explosions:
                explosion.update(dt)
            self.explosions = [e for e in self.explosions if not e.discard]
        
        
        update_music_stream(self.audio['music'])

    def draw(self):
        begin_drawing()
        clear_background(BG_COLOR)
        
        if not self.game_over:
            self.draw_stars()
            self.draw_score()
            self.draw_health()  # Draw health indicators
            self.draw_timer()   
            self.player.draw()
            
            # Draw all game objects
            for sprite in self.lasers + self.meteors + self.enemies + self.enemy_lasers + self.explosions:
                sprite.draw()
            # Draw beams
            for beam in self.enemy_beams:
                beam.draw()

            # Draw wave information
            self.wave_manager.draw()

            # Special drawing for enemies to show health
            for enemy in self.enemies:
                enemy.draw()
                
                # Draw health indicators for multi-hit enemies
                if hasattr(enemy, 'health') and enemy.health > 1:
                    # Draw small rectangles above enemy showing remaining health
                    center_x = enemy.pos.x + enemy.size.x / 2
                    for i in range(enemy.health):
                        draw_rectangle(
                            int(center_x - enemy.health * 5 + i * 10), 
                            int(enemy.pos.y - 10),
                            8, 4, RED
                        )
             # Add before end_drawing():
            self.draw_weapon_info()
        else:
            # Draw game over screen
            self.draw_game_over()
            
        end_drawing()

    # Update the run method to handle game over state
    def run(self):
        while not window_should_close():
            # Check for game restart
            if self.game_over and is_key_pressed(KEY_ENTER):
                # Reset the game state
                self.reset_game()
            
            self.update()
            self.draw()
        
        unload_music_stream(self.audio['music'])
        close_audio_device()
        close_window()


    # Add a game over screen
    def draw_game_over(self):
        self.draw_stars()  # Keep drawing background
        
        game_over_text = "GAME OVER"
        score_text = f"Final Score: {self.score}"
        restart_text = "Press ENTER to restart"
        
        # Measure text sizes
        game_over_size = measure_text_ex(self.assets['font'], game_over_text, FONT_SIZE * 2, 0)
        score_size = measure_text_ex(self.assets['font'], score_text, FONT_SIZE, 0)
        restart_size = measure_text_ex(self.assets['font'], restart_text, FONT_SIZE, 0)
        
        # Draw text centered
        draw_text_ex(self.assets['font'], game_over_text, 
                    Vector2((WINDOW_WIDTH / 2) - (game_over_size.x / 2), (WINDOW_HEIGHT / 2) -(game_over_size.y / 2) ), 
                    FONT_SIZE * 2, 0, RED)
        
        draw_text_ex(self.assets['font'], score_text, 
                    Vector2((WINDOW_WIDTH / 2) - (score_size.x / 2), (WINDOW_HEIGHT / 2) +(score_size.y / 2) ), 
                    FONT_SIZE, 0, WHITE)
        
        draw_text_ex(self.assets['font'], restart_text, 
                    Vector2((WINDOW_WIDTH / 2) - (restart_size.x / 2), (WINDOW_HEIGHT / 2) + 200), 
                    FONT_SIZE, 0, WHITE)
        
    def reset_game(self):
        # Clear all game objects
        self.lasers.clear()
        self.meteors.clear()
        self.enemies.clear()
        self.enemy_lasers.clear()
        self.explosions.clear()
        
        # Reset player
        self.player = Player(self.assets['player'], 
                            Vector2(WINDOW_WIDTH / 2, WINDOW_HEIGHT / 2), 
                            self.shoot_laser)
        
        # IMPORTANT: Reinitialize weapons for the new player
        from weapons import Weapon, TripleShot, RapidFire
        self.player.weapons = [
            Weapon(self),          # Standard laser
            TripleShot(self),      # Triple shot 
            RapidFire(self)        # Rapid fire
        ]
        
        # Keep weapons unlocked based on current wave
        if hasattr(self, 'wave_manager'):
            if self.wave_manager.current_wave >= 2:
                self.player.weapons[1].unlocked = True
            if self.wave_manager.current_wave >= 3:
                self.player.weapons[2].unlocked = True
        
        # Reset score and game state
        self.score = 0
        self.game_over = False

        # Reset wave manager
        self.wave_manager.reset()
        
        # Reset enemy timer (important!)
        self.enemy_timer.deactivate()  # Stop current timer
        self.enemy_timer.duration = ENEMY_SPAWN_INTERVAL  # Reset to default duration

if __name__ == '__main__':
    game = Game()
    game.run()